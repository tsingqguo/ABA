from . import BaseActor
import torch.nn as nn
import torch
#from attack.pgd import IPGD , IPGD_after,IPGD_siamkys,IPGD_siamrpn
import cv2
from extern.stark.utils.box_ops import box_xyxy_to_cxcywh ,box_cxcywh_to_xyxy,box_xywh_to_xyxy,box_cxcywh_to_xywh,giou_loss
from extern.stark.utils.misc import NestedTensor
from extern.stark.utils.merge import merge_template_search

import torch.nn.functional as F
from ltr.models.kys.utils import DiMPScoreJittering
import numpy as np
from  visdom import Visdom
vis=Visdom(env="rnn")
#from ltr.models.pysot.models.loss import select_cross_entropy_loss

attack_model = False
def save_torchimg(img,name, anno=None):
    
    save_im = img[0].detach().permute(1,2,0).cpu().numpy().copy()
    if anno is not None:
        bbox = list(map(int, anno*320))
        #print(bbox,save_im.shape)
        cv2.rectangle(save_im, (bbox[0], bbox[1]),
                (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
    cv2.imwrite('/workspace/ABA/debug/train/{}.png'.format(name), save_im)

class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats

class KYSActor(BaseActor):
    """ Actor for training KYS model """
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first test frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[-2],
                                                                                             backbone_feat_cur.shape[-1])
            else:
                motion_feat_cur = backbone_feat_cur

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            # Jitter target model output for augmentation
            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            # Input target model output along with previous frame information to the predictor
            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats

class SIAM_KYSActor(BaseActor):
    """ Actor for training KYS model """
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_fn = dimp_jitter_fn
        self.attacker = IPGD_siamrpn()
        # TODO set it somewhere
        self.device = torch.device("cuda:0")
        self.maxpool = nn.MaxPool2d(25)


    def log_det(self, y_true, y_pred, num_model=3):
        cat_pred = torch.cat((y_pred[0],y_pred[1],y_pred[2]),dim=1)
        map_shape = cat_pred.shape
        y_true = y_true.permute(1,0,2,3)
        bool_R_y_true = torch.ne(torch.ones_like(y_true) - y_true, 0) # batch_size X (num_class X num_models), 2-D
        #print(cat_pred.size(),bool_R_y_true.size())
        mask_non_y_pred = torch.masked_select(cat_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D  mask为True 的元素值 此处为预测真是类以外的数值
        #print(mask_non_y_pred.size())
        
        mask_non_y_pred = mask_non_y_pred.reshape(map_shape).view(map_shape[0],num_model,-1)
        #print(mask_non_y_pred.size())
        
        #mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D   变换维度
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, dim=2, keepdim=True) # batch_size X num_model X (num_class-1), 3-D   标准化
        matrix = torch.bmm(mask_non_y_pred, mask_non_y_pred.permute(0,2,1)) # batch_size X num_model X num_model, 3-D   矩阵乘
        all_log_det = torch.det(matrix + 1e-6*(torch.eye(num_model).unsqueeze(0).cuda())).log() # batch_size X 1, 1-D  tf.eye对角矩阵3x3   计算matrix行列式的对数.
        #print(all_log_det.size())
        all_log_det = - all_log_det.sum().item()/map_shape[0]
        return all_log_det

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]
        test_z = data['test_z'].to(self.device).permute(1,0,2,3) 
        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)
        #print(data['test_images'].shape,data['test_valid_image'].shape)  #torch.Size([50, 10, 3, 288, 288]) torch.Size([50, 10]) 50帧 ，10个视频
    
        
        # Initialize loss variables
        my_loss = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames
        #print(test_z.size())
        #save_torchimg(test_z,'zff')
        #print(test_z)
        
        #print(data['test_images'][0])
        zf = self.net.backbone(test_z)
        zf_all = self.net.neck(zf)
        
        xf = self.net.backbone(data['test_images'][0].to(self.device)*255)
        xf_prev = self.net.neck(xf)
        xf_prev = self.net.adjcon1(xf_prev[2])
        xf_prev = self.net.adjcon2(xf_prev)
        #print(xf_prev.size())
        
        #dimp_filters = self.net.train_classifier(train_images, train_anno)
        #print(train_images.size())  #torch.Size([3, 10, 3, 288, 288])
        #print(dimp_filters.size())
        
        # Track in the first test frame
        #test_image_cur = data['test_images'][0, ...].to(self.device)
        #backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        #print(backbone_feat_prev_all.keys())
        
        '''backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()'''
       
        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        #print(data['test_anno'].size(),data['test_label'].size())   #torch.Size([50, 10, 4]) torch.Size([50, 10, 19, 19])
        
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length): #按每帧循环，每个循环同时载入三个视频的对应帧
            #print(i)
            test_image_cur = data['test_images'][i, ...].to(self.device)*255
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)
            #print(test_image_cur)
            
            # Extract features
            #print(test_image_cur.size())  #torch.Size([10, 3, 288, 288])
            #print(self.net.training)
            #print('pass')
            #if self.net.training:
                #print('adv')
            if i >20:
                
                mydata={'zf': zf_all,
                        'label_prev': label_prev, 
                        'anno_prev': anno_prev,
                        'state_prev':state_prev,
                        'xf_prev':xf_prev
                        }
                test_image_cur = self.attacker.attack(self.net, test_image_cur, mydata)
                
            
            zf = zf_all
            xf = self.net.backbone(test_image_cur)
            #print(xf[0].size(),xf[1].size(),xf[2].size())
            xf = self.net.neck(xf)
            #print(xf[0].size(),xf[1].size(),xf[2].size())
            #feat2 = self.net.maxpool(xf[2])
            #print(feat2.size())
            
            feat2 = self.net.adjcon1(xf[2])
            feat2 = self.net.adjcon2(feat2)
            cls, loc = self.net.rpn_head(zf, xf) # B , 10 , 25 , 25
            #bats = cls.size(0)
            
            b, a2, h, w = cls.size()
            cls = cls.view(b, 2, a2//2, h, w)
            cls = cls.permute(0, 2, 3, 4, 1).contiguous()
            cls = F.softmax(cls, dim=4)[:,:,:,:,1]
            #print(cls.size())  #torch.Size([6, 5, 25, 25])
            
            _ , maxid = torch.max(cls.view(b, -1),dim=1)
            maxid = maxid/625
            #print(maxid)

            
            score_ls=[]
            for ss in range(b):
                score_ls.append(cls[ss,maxid[ss],:,:])
            score_one = torch.stack(score_ls).unsqueeze(1) 
            #print(score_one.size())
            
            #for i in range(5):
                #vis.heatmap(cls[0,i,:,:])
            
            '''score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
            score = F.softmax(score, dim=1).data[:, 1].view(bats, 5,25,25)
            
            score_maxid= self.maxpool(score).view(bats,-1)
            #score_map = score.sum(1)/5
            #print(score_maxid.size())
            #print(score_maxid)
            _ , maxid = torch.max(score_maxid,dim=1)
            #print(maxid)
            score_ls=[]
            for i in range(len(maxid)):
                
            score_one = torch.stack(score_ls)'''
         
        
            
            rnn_data={
                'feat1':xf_prev ,
                'feat2':feat2,
                'dimp_score_cur':score_one,
                'state_prev': state_prev,
                'label_prev':label_prev
            }


            #print(len(xf))
            
            predictor_output = self.net.RNN_defence(rnn_data)

            #predictor_output = self.net.predictor(predictor_input_data)
            
            predicted_resp = predictor_output['response']#.unsqueeze(0) 
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']
            #fgr   使用siamrpn损失函数，调整PGD ，auxiliary_outputs 的 GT size  test_label_cur
            #vis.heatmap(score_one[0][0])
            #vis.heatmap(predicted_resp[0][0])
            #vis.heatmap(test_label_cur[0][0])
            #input()
            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            #print(is_valid)  #一堆  1
            
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame
            #ee_loss = aux_data['ee_loss']
            #log_det_loss = self.log_det(test_label_cur,aux_data['pred_list'])
            #print(ee_loss,log_det_loss)

            #my_loss[:, i - 1] = 0.002 * ee_loss + 0.05 * log_det_loss
            # Calculate losses   anno是bbox  label 是 map
            #print(predicted_resp.size(), data['cls_label'][i].size())
            
            #label_cls = data['cls_label'][i]#.cpu().numpy() 
            '''print(predicted_resp.size(),  test_label_cur.size())
            vis.heatmap(predicted_resp.sum(1).unsqueeze(0)[0][0])
            vis.heatmap(predicted_resp.sum(1).unsqueeze(0)[0][1])
            vis.heatmap(predicted_resp.sum(1).unsqueeze(0)[0][2])

            vis.heatmap(test_label_cur[0][0])
            vis.heatmap(test_label_cur[0][1])
            vis.heatmap(test_label_cur[0][2])'''
            test_label_cur = test_label_cur.permute(1,0,2,3)
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,     #map的  MSE loss    .sum(1).unsqueeze(0)/5
                                                           test_anno_cur, valid_samples=is_valid)
                                                      
            #cls = self.log_softmax(predicted_resp)
            #predicted_resp = predicted_resp.detach().log()#.cpu().numpy()
            #clf_loss_test_new =  torch.Tensor([select_cross_entropy_loss(predicted_resp, label_cls) ] ) 
            #print(clf_loss_test_new)  
                                  
            #print(clf_loss_test_new.size())
                                                          
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()


            '''dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()'''

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():  #fused_score_orig 经过 (dimp_score_cur > dimp_thresh)和window = predicted_resp
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].sum(1).unsqueeze(0).view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()
            
            # is_target = self.is_target_predictor(state_prev)
            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            # is_target_after_prop = self.is_target_predictor(propagated_h)
            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            '''print(aux_data['is_target'].size())
            
            vis.heatmap(aux_data['is_target'][0][0])
            vis.heatmap(aux_data['is_target'][1][0])
            vis.heatmap(aux_data['is_target'][2][0])

            vis.heatmap(label_prev[0][0])
            vis.heatmap(label_prev[0][1])
            vis.heatmap(label_prev[0][2])'''
            

            '''test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct'''

            #dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            #dimp_clf_acc += dimp_clf_acc_new

            #dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            xf_prev = feat2.clone()
            #dimp_scores_prev = score_one.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean() 
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        #dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        #dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test  # 0

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss# 50
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w + my_loss.mean()

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 #'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 #'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 #'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 #'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 #'myloss': my_loss.mean()
                 }

        return loss, stats

class STARK_STAFActor(BaseActor):
    """ Actor for training KYS model """
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_fn=None,settings=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.dimp_jitter_fn = dimp_jitter_fn
        #self.attacker = IPGD_siamrpn()
        # TODO set it somewhere
        self.device = torch.device("cuda:0")
        self.maxpool = nn.MaxPool2d(25)
        

    def log_det(self, y_true, y_pred, num_model=3):
        cat_pred = torch.cat((y_pred[0],y_pred[1],y_pred[2]),dim=1)
        map_shape = cat_pred.shape
        y_true = y_true.permute(1,0,2,3)
        bool_R_y_true = torch.ne(torch.ones_like(y_true) - y_true, 0) # batch_size X (num_class X num_models), 2-D
        #print(cat_pred.size(),bool_R_y_true.size())
        mask_non_y_pred = torch.masked_select(cat_pred, bool_R_y_true) # batch_size X (num_class-1) X num_models, 1-D  mask为True 的元素值 此处为预测真是类以外的数值
        #print(mask_non_y_pred.size())
        
        mask_non_y_pred = mask_non_y_pred.reshape(map_shape).view(map_shape[0],num_model,-1)
        #print(mask_non_y_pred.size())
        
        #mask_non_y_pred = tf.reshape(mask_non_y_pred, [-1, num_model, num_classes-1]) # batch_size X num_model X (num_class-1), 3-D   变换维度
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, dim=2, keepdim=True) # batch_size X num_model X (num_class-1), 3-D   标准化
        matrix = torch.bmm(mask_non_y_pred, mask_non_y_pred.permute(0,2,1)) # batch_size X num_model X num_model, 3-D   矩阵乘
        all_log_det = torch.det(matrix + 1e-6*(torch.eye(num_model).unsqueeze(0).cuda())).log() # batch_size X 1, 1-D  tf.eye对角矩阵3x3   计算matrix行列式的对数.
        #print(all_log_det.size())
        all_log_det = - all_log_det.sum().item()/map_shape[0]
        return all_log_det

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def attack_stark(self, net, search_img, search_att, feat_dict_tem, feat1, tem_anno,prev_bbox):
        
        bs = search_img.size(0)
        pert = torch.zeros(search_img.size()).cuda()
        adv_cls = torch.zeros(bs,2,20,20).cuda()
        adv_point = np.random.randint(1,18,[4*bs])
    
        for d in range(bs):
            adv_cls[d,0,adv_point[d*4+0],adv_point[d*4+1]] = 10
            adv_cls[d,1,adv_point[d*4+2],adv_point[d*4+3]] = 10
   
        for aa in range(8):
            pert = pert.detach()
            pert.requires_grad = True
            search_img = search_img.detach()
            adv_cls = adv_cls.detach()
            search_att = search_att.detach()
            if prev_bbox is not None :
                prev_bbox = prev_bbox.detach()
            #feat_dict_tem = feat_dict_tem.detach()
            feat1 = feat1.detach()
            tem_anno = tem_anno.detach()
            feat_list = []
            feat_dist = self.net(img=NestedTensor(search_img +pert, search_att), mode='backbone')
            feat2 = feat_dist['ori_feat']
       
            feat_list.append(feat_dict_tem)
            feat_list.append(feat_dist)  
            seq_dict = merge_template_search(feat_list)

            data_rnn={
                'feat1':feat1,
                'feat2':feat2,
                'tem_anno':tem_anno,
                'use_staf':True,
                'update': False,
                'prev_lab':prev_bbox
            }
            out_dict, _, _ = net.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=False, data=data_rnn)
            act_map = out_dict['auxiliary']['fused_score_orig']
            #vis.heatmap(act_map[0][0][0])
            #vis.heatmap(act_map[1][0][0])
            loss = torch.norm((act_map[0]-adv_cls[:,0,:,:].unsqueeze(1) + act_map[1]-adv_cls[:,1,:,:].unsqueeze(1)),1).sum()
            #print(loss)
            loss.backward()
            x_grad = -pert.grad
            x_grad = torch.sign(x_grad)
            adv_x = search_img + pert + 3/255 * x_grad
            pert = adv_x - search_img
            pert = torch.clamp(pert, -20/255, 20/255)

        return pert + search_img
        

    def __call__(self, data):
        data['train_images'] = data['search_images']
        # process the templates
        #print(data['template_images'].size(),data['search_images'].size(),data['search_anno'].size(),data['template_big'].size())  
        #torch.Size([1, 4, 3, 128, 128]) torch.Size([10, 4, 3, 320, 320]) torch.Size([10, 4, 4]) torch.Size([1, 4, 3, 320, 320])
        num_sequences = data['search_images'].size(1)
        sequence_length = data['search_images'].size(0)
        loss_actmap_all = torch.zeros(num_sequences, sequence_length ).to(self.device)
        loss_state_all = torch.zeros(num_sequences, sequence_length ).to(self.device)
        loss_box_all = torch.zeros(num_sequences, sequence_length ).to(self.device)
        #print(data['bbox_tembig'][0].size())
        
        #vis.image(data['template_images'][:,0,:,:,:])
        #save_torchimg(data['template_images'][:,0,:,:,:]*255,'tem')
        #save_torchimg(data['search_images'][0,0,:,:,:].unsqueeze(0)*255,'search')
        #for i in range(32):
            #save_torchimg(data['template_big'][:,i,:,:,:]*255,'tem_big{}'.format(i),data['bbox_tembig'][0][i])
        #save_torchimg(data['template_big'][:,0,:,:,:]*255,'tem_big')
        #save_torchimg(data['template_images'][:,0,:,:,:]*255,'tem')
        #写一个可视化函数，看gt是否正确

        '''for i in range(self.settings.num_search):
            save_torchimg(data['search_images'][i,0,:,:,:].unsqueeze(0)*255,'search{}'.format(i),data['search_anno'][i][0])
            print(data['search_anno'][i][0])'''
        
        #
        #print(data['search_anno'][0])   #以search 左上角为起点 cx ,cy ,w,h 数值为比例，即0-1
        template_img_i = data['template_images'][0].view(-1, *data['template_images'].shape[2:]).to(self.device)  # (batch, 3, 128, 128)
        template_att_i = data['template_att'][0].view(-1, *data['template_att'].shape[2:]).to(self.device)  # (batch, 128, 128)
        template_big = data['template_big'][0].to(self.device)
        tem_anno = data['bbox_tembig'][0].to(self.device)
        #print(tem_anno)
        feat1 ,_= self.net.backbone(NestedTensor(template_big.float(), template_att_i))
        #print(feat1[0].size())
        feat1 ,_= feat1[-1].decompose()
        feat_dict_tem = self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone')
      
        prev_bbox =None
        # process the search regions (t-th frame)
        for i in range(self.settings.num_search):
            feat_dict_list = []
            #print(data.keys())
            search_anno = data['search_anno'][i].to(self.device)
            bs = search_anno.size(0)
            #print('sss',search_anno)
            search_img = data['search_images'][i].view(-1, *data['search_images'].shape[2:]).to(self.device) # (batch, 3, 320, 320)
            search_att = data['search_att'][i].view(-1, *data['search_att'].shape[2:]).to(self.device)  # (batch, 320, 320)

            if attack_model:
                search_img = self.attack_stark(self.net,search_img, search_att,feat_dict_tem,feat1,tem_anno,prev_bbox)

            
            feat_dist = self.net(img=NestedTensor(search_img, search_att), mode='backbone')
            feat2 = feat_dist['ori_feat']
            #print(feat2.size())
            feat_dict_list.append(feat_dict_tem)
            feat_dict_list.append(feat_dist)  
            
            #save_torchimg(search_img[0].unsqueeze(0)*255,'search')
            #print(feat_dict_list[0])
            #print(type(feat1),type(feat2))
            
            # run the transformer and compute losses
            seq_dict = merge_template_search(feat_dict_list)
            data_rnn={
                'feat1':feat1,
                'feat2':feat2,
                'tem_anno':tem_anno,
                'use_staf':True,
                'update': True,
                'prev_lab':prev_bbox
            }
            out_dict, _, _ = self.net.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=False, data=data_rnn)
            
            #print(out_dict['pred_boxes'].size())
            
            search_lable_xyxy = box_xywh_to_xyxy(search_anno).clamp(0,1) *20
            search_lable_cxcywh = box_xyxy_to_cxcywh(search_lable_xyxy/20).clamp(0,1) 
            #print('ttt',search_lable_xyxy)
            kx = cv2.getGaussianKernel(39,1)
            ky = cv2.getGaussianKernel(39,1)
            gaus =  np.multiply(kx,np.transpose(ky)) * 9
            gauss_map = torch.from_numpy(gaus).cuda().float()#.repeat(bs,1,1)  # bs  ,40,40
            #print(label_gauss_map.size())
            label_gauss_map = torch.zeros(bs,20,20).cuda()
            label_gauss_map_copy = torch.zeros(bs,20,20).cuda()
     
            for dd in range(bs):
                xx = int(np.around(search_lable_xyxy[dd,0].cpu().numpy().clip(0,18)))
                yy = int(np.around(search_lable_xyxy[dd,1].cpu().numpy().clip(0,18)))
                xx2 = int(np.around(search_lable_xyxy[dd,2].cpu().numpy().clip(0,18)))
                yy2 = int(np.around(search_lable_xyxy[dd,3].cpu().numpy().clip(0,18)))
                #print(xx,yy)
                
                label_gauss_map[dd] = gauss_map[18-yy:38-yy,18-xx:38-xx]
                label_gauss_map_copy[dd] = gauss_map[18-yy2:38-yy2,18-xx2:38-xx2]
            label_gauss_map_mix = (label_gauss_map + label_gauss_map_copy).unsqueeze(1) 
            #label_gauss_map = out_dict['auxiliary']['score_mix']#[:,0,:,:]
            #print(label_gauss_map.size())
            #prev_bbox = out_dict['pred_boxes']  #cxcywh 
            cxcywh_box = out_dict['pred_boxes'].view(-1,4)
            prev_bbox = box_cxcywh_to_xywh(cxcywh_box)
            xyxy_bbox = box_cxcywh_to_xyxy(cxcywh_box)
            #print(pred_box,search_lable_xyxy/20)
            
            act_map = out_dict['auxiliary']['fused_score_orig']#[:,0,:,:]
            state_after = out_dict['auxiliary']['is_target_after_prop']#[:,0,:,:] #  is_target   is_target_after_prop
            #print(act_map.size())
            
            feat1 = feat2.detach()
            #print(xyxy_bbox.size(),search_lable_xyxy.size())
            #print(xyxy_bbox - search_lable_xyxy/20)
            '''try:
                iou_box , _=  self.objective['giou'](xyxy_bbox , search_lable_xyxy/20)
            except:
                iou_box = torch.tensor(0.0).cuda()'''
            #print(act_map.size())
            map_1 = act_map[0]#[:,0,:,:].view(-1,1,20,20)
            map_2 = act_map[1]#[:,1,:,:].view(-1,1,20,20)
            #vis.heatmap(map_2[0][0])
            #for sd in range(bs):
            max_v1 = torch.max(map_1.view(-1))
            max_v2 = torch.max(map_2.view(-1))
            label_gauss_map = label_gauss_map * max_v1
            label_gauss_map_copy = label_gauss_map_copy * max_v2
            #vis.heatmap(label_gauss_map[0])
            
            li_box =  self.objective['l1'](xyxy_bbox , search_lable_xyxy/20)
            l1_actmap = self.objective['LBHinge'](map_1,label_gauss_map.unsqueeze(1)) + self.objective['LBHinge'](map_2,label_gauss_map_copy.unsqueeze(1))
            l1_state = self.objective['LBHinge'](state_after,label_gauss_map_mix)
            #print(l1_actmap,l1_state)
            #print(iou_box , self.loss_weight['iou'])
            loss = l1_actmap  * self.loss_weight['act_map']  + l1_state *self.loss_weight['state_after_prop'] + li_box * self.loss_weight['iou']
            loss_actmap_all[:, i ] = l1_actmap
            loss_state_all[:, i ] = l1_state
            loss_box_all[:, i ] = li_box
        loss_box_all = loss_box_all.mean()
        loss_actmap_all = loss_actmap_all.mean()
        loss_state_all = loss_state_all.mean()
        loss = loss/sequence_length
        status = {"Loss/total": loss.item(),
                    "Loss/actmap": loss_actmap_all.item(),
                    "Loss/state": loss_state_all.item(),
                    "Loss/box": loss_box_all.item()
                    }
                      

        return loss, status