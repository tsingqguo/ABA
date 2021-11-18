from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.core.config2 import cfg2
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from extern.pytracking.features.preprocessing import numpy_to_torch
from pysot.utils.bbox import get_axis_aligned_bbox
from extern.pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from attacker_dimp import attacker
from attacker_kys import attacker as attacker_kys 
from attacker import attacker as attacker_siam
from  visdom import Visdom
vis=Visdom(env="attack_test")


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--testid', default='1', type=str,
        help='测试序号')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)
fast_test_all = False

a0 = 32
a1 = 33
a2 = 36
img_num = 5
imlist=[]
test_mode = 'testall'   #   debug    testall  visionall  transfer  transfer_kys   transfer_siam
attack_mode = 'guo_fine_speed'    #     guo  nor_blur    whitebox     obj_speed 
attack_bool = True
tran_path = '/workspace/ABA/experiments/dimp_50/'  #攻击源   siamrpn_r50_l234_dwxcorr    siamrpn_mobilev2_l234_dwxcorr



def get_move_dis(box0,box1):
    cx0 = box0[0]+0.5*box0[2]
    cy0 = box0[1]+0.5*box0[3]
    cx1 = box1[0]+0.5*box1[2]
    cy1 = box1[1]+0.5*box1[3]
    d_mov = ((cx0-cx1)**2+(cy0-cy1)**2)** 0.5
    return d_mov

def save_torchimg(img,name):
    save_im = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/workspace/ABA/compare_img/{}.jpg'.format(name), save_im)

def main():
    # load config
    cfg.merge_from_file(args.config)
    vis_img = False
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    model = None
    '''
    for name,parameters in model.named_parameters():
            print(name,':',parameters.size())
    '''
    # build tracker
    tracker = build_tracker(model)
    #tracker_att = build_tracker(model)
    ack = attacker()
    
    ack_kys = attacker_kys()
    if test_mode == 'transfer' or test_mode == 'transfer_kys':  # tracker_att 是生成攻击样本的tracker
        cfg.merge_from_file(tran_path+'config.yaml')
        tran_tracker = build_tracker(model)

    if test_mode == 'transfer_siam':
        cfg.merge_from_file(tran_path + 'config.yaml')
        tran_model = ModelBuilder()
        tran_model = load_pretrain(tran_model, tran_path + 'model.pth').cuda().train()
        tran_tracker = build_tracker(tran_model)
        ack_siam = attacker_siam()
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    
    model_name = 'opaba' #args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019'] : #and not os.path.exists(os.path.join('attack', args.dataset, model_name)):  #,'baseline'
        print(cfg.PYTRACKING)
        input()
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            if os.path.exists(os.path.join('/workspace/ABA/experiments/dimp_50/results', args.dataset, model_name,'baseline',video.name)):
                print()
                #continue
            print(video.name)
            
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                
                if idx==a2:
                    im2 = img
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    if 'transfer' in test_mode :
                        tran_tracker.init(img, gt_bbox_)
                    tracker.init(img, gt_bbox_)
                    # tracker_att.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    X0 = img.copy()
                elif idx > frame_counter:
                    #tracker_att.track(img)
                    if idx %5 == 0 and attack_bool and test_mode == 'testall':
                        

                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True)
                        #print(img_attack.size())
                        #outputs = tracker.track(img,x_crop = ispert=True)
                        outputs = tracker.track(img, x_crop = out_ack['adv_x_crop'],coord = out_ack['coord'],ispert=True)
                    
                        
                        
                        
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        vis_img = True
                        '''
                        if overlap < 0.2:
                            print(overlap)
                            vis.image(out_ack['adv_x_crop'][0])
                            input()
                        '''
                        
                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer':
                        X1=img.copy()
                        #print(X0.shape)
                        X0_t = numpy_to_torch(X0)
                        X1_t = numpy_to_torch(X1)
                        x0_patches, _ = sample_patch_multiscale(X0_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        x1_patches, pos = sample_patch_multiscale(X1_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        #print(x0_patches.size())
                        #save_torchimg(x1_patches,'dimp_tran')
                        
                        out_ack = ack(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tran_tracker , speed = True ,crop = [x0_patches,x1_patches])
                        #print(img_attack.size())
                        #save_torchimg(out_ack['adv_x_crop'],'trans')
                        outputs = tracker.track(img,  x_crop  = out_ack['adv_x_crop'],coord =  pos ,ispert=True)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()    

                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer_siam':
                        X1=img.copy()
                        #print(X0.shape)
                        X0_t = numpy_to_torch(X0)
                        X1_t = numpy_to_torch(X1)
                        x0_patches, _ = sample_patch_multiscale(X0_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        x1_patches, pos = sample_patch_multiscale(X1_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        #print(x1_patches.size())
                        x1_patches = torch.nn.functional.interpolate(x1_patches, size=(cfg.TRACK.INSTANCE_SIZE,cfg.TRACK.INSTANCE_SIZE), mode='bilinear')
                        x0_patches = torch.nn.functional.interpolate(x0_patches, size=(cfg.TRACK.INSTANCE_SIZE,cfg.TRACK.INSTANCE_SIZE), mode='bilinear')
                        
                        out_ack = ack_siam(img0=X0,img1=X1, mode = 'guo_crop_speed', model_attacked = tran_tracker ,speed = True ,crop = [x0_patches,x1_patches])
                        #print(img_attack.size())                  guo_crop_speed
                        #save_torchimg(out_ack['adv_x_crop'],'trans')
                        #print(out_ack['adv_x_crop'].size(),tracker.img_sample_sz)
                        dimp_sz = tracker.img_sample_sz[0].int().item()
                        #print(dimp_sz)
                        resize_blur  = torch.nn.functional.interpolate(out_ack['adv_x_crop'], size=(dimp_sz,dimp_sz), mode='bilinear')
                        #print(resize_blur.size())
                        outputs = tracker.track(img,  x_crop  = resize_blur ,coord = pos ,ispert=True)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input() 

                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer_kys':
                        X1=img.copy()
                        #print(X0.shape)
                        X0_t = numpy_to_torch(X0)
                        X1_t = numpy_to_torch(X1)
                        x0_patches, _ = sample_patch_multiscale(X0_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker_att.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        x1_patches, cood = sample_patch_multiscale(X1_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker_att.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        #print(x0_patches.size())
                        #save_torchimg(x1_patches,'dimp_tran')
                        
                        out_ack = ack_kys(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tracker_att, speed = True ,crop = [x0_patches,x1_patches])
                        #print(img_attack.size())
                        #save_torchimg(out_ack['adv_x_crop'],'trans')
                        outputs = tracker.track(img,  x_crop  = out_ack['adv_x_crop'], coord = cood , ispert=True)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()    
                    else:
                        #print(img.shape)
                        X0 = img.copy()
                        outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter and vis_img:
                    print(idx)
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    #pred_bbox_normblur = list(map(int, pred_bbox_normblur))
                    #pred_bbox_advblur  = list(map(int, pred_bbox_advblur))
                    
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 1)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (255, 255, 0), 1)
                    
                    #cv2.rectangle(out_ack['normblur_org'], (pred_bbox_normblur[0], pred_bbox_normblur[1]),
                                  #(pred_bbox_normblur[0]+pred_bbox_normblur[2], pred_bbox_normblur[1]+pred_bbox_normblur[3]), (255, 255, 0), 1)
                    
                    #cv2.rectangle(out_ack['advblur_org'], (pred_bbox_advblur[0], pred_bbox_advblur[1]),
                                  #(pred_bbox_advblur[0]+pred_bbox_advblur[2], pred_bbox_advblur[1]+pred_bbox_advblur[3]), (0, 255, 255), 1)
                    
                    
                    cv2.putText(img, str(idx)+'original', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    #cv2.putText(out_ack['normblur_org'], 'normblur', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    #cv2.putText(out_ack['advblur_org'], 'advblur', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    vis.image(img.transpose(2,0,1),win='img')
                    vis.image(out_ack['adv_x_crop'][0],win='ack')
                    #vis.image(out_ack['normblur_org'].transpose(2,0,1))
                    
                    #vis.image(out_ack['advblur_org'].transpose(2,0,1))
                    vis_img = False
                    #cv2.imshow(video.name, img)
                    print('按一下')
                    input()
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            
            total_lost += lost_number
            
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else  :
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            if os.path.exists(os.path.join('/workspace/ABA/experiments/dimp_50/results/LaSOT',model_name,video.name+'.txt')):
                print('pass ' + video.name)
                continue
            #print((os.path.join('/workspace/ABA/experiments/dimp_50/results/UAV/guo_speed_grad_norm',video.name+'.txt')))
            
            print('Now is running ' + video.name)
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            d_m = 0
            box1 =None
            print('Processing video:', video.name, model_name)
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]

                    tracker.init(img, gt_bbox_)
                    if 'transfer' in test_mode :
                        tran_tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                    box0 = pred_bbox
                    X1 = img.copy()
                else:
                    
                    if idx==a0:
                        im0 = img
                    if idx==a1:
                        im1 = img
                    
                    if test_mode == 'debug' and idx==a1 and attack_bool:   #   ,X3=img,
                        imlist=None
                        img_attack = ack(img0=im0,img1=im1,mode = attack_mode ,model_attacked = tracker,tracker = tracker)
                        outputs = tracker.track(img, v=1 , x_c = img_attack['adv_x_crop'])
                        sf
                    elif idx %5 == 0 and attack_bool and test_mode == 'testall':
                        
                        #X0 = im0
                        
                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True )
                        #print(img_attack.size())
                        #outputs = tracker.track(img,x_crop = ispert=True)
                        
                        outputs = tracker.track(img, x_crop = out_ack['adv_x_crop'],coord = out_ack['coord'],ispert=True)
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()
                    
                    elif idx %5 == 0 and attack_bool and test_mode == 'saveimg':
                        
                        #X0 = im0
                        
                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True )
                        #print(img_attack.size())
                        #outputs = tracker.track(img,x_crop = ispert=True)
                        
                        outputs = tracker.track(img, x_crop = out_ack['adv_x_crop'],coord = out_ack['coord'],ispert=True)
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)
                        input()
                        
                    elif idx %5 == 0 and test_mode == 'visionall' and attack_bool:
                        
                        imlist=None
                        
                        X1=img.copy()
                        out_ack = ack(X0,X1,mode = attack_mode ,model_attacked = tracker)
                        #print(out_ack['x_crop_normblur'].size())
                        #vis.image(img.transpose(2,0,1))
                        pred_bbox_normblur = tracker.track(out_ack['normblur_org'] ,   update=False)['bbox']
                        pred_bbox_advblur  = tracker.track(img, x_c = out_ack['adv_x_crop']     , update=False)['bbox']
                        
                        pred_bbox          = tracker.track(img,  v =False ,update=True)['bbox']
                        overlap_adv= vot_overlap(pred_bbox, pred_bbox_advblur, (img.shape[1], img.shape[0]))
                        print(overlap_adv)
                        
                        if overlap_adv < 0.4:
                            vis_img =True
                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer': #dimp 互相攻击
                        X1=img.copy()
                        #print(X0.shape)
                        X0_t = numpy_to_torch(X0)
                        X1_t = numpy_to_torch(X1)
                        x0_patches, _ = sample_patch_multiscale(X0_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        x1_patches, pos = sample_patch_multiscale(X1_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        out_ack = ack(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tracker_att,crop = [x0_patches,x1_patches])
                        #print(img_attack.size())
                        #save_torchimg(out_ack['adv_x_crop'],'trans')
                        outputs = tracker.track(img,  x_crop  = out_ack['adv_x_crop'],coord = pos ,ispert=True)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()    

                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer_kys':
                        X1=img.copy()
                        #print(X0.shape)
                        X0_t = numpy_to_torch(X0)
                        X1_t = numpy_to_torch(X1)
                        x0_patches, _ = sample_patch_multiscale(X0_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker_att.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        x1_patches, cood = sample_patch_multiscale(X1_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker_att.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        #print(x0_patches.size())
                        #save_torchimg(x1_patches,'dimp_tran')
                        
                        out_ack = ack_kys(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tracker_att, speed = True ,crop = [x0_patches,x1_patches])
                        #print(img_attack.size())
                        #save_torchimg(out_ack['adv_x_crop'],'trans')
                        outputs = tracker.track(img,  x_crop  = out_ack['adv_x_crop'], coord = cood , ispert=True)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()    

                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer_siam':
                        X1=img.copy()
                        #print(X0.shape)
                        X0_t = numpy_to_torch(X0)
                        X1_t = numpy_to_torch(X1)
                        x0_patches, _ = sample_patch_multiscale(X0_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        x1_patches, pos = sample_patch_multiscale(X1_t, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
                        #print(x1_patches.size())
                        x1_patches = torch.nn.functional.interpolate(x1_patches, size=(cfg.TRACK.INSTANCE_SIZE,cfg.TRACK.INSTANCE_SIZE), mode='bilinear')
                        x0_patches = torch.nn.functional.interpolate(x0_patches, size=(cfg.TRACK.INSTANCE_SIZE,cfg.TRACK.INSTANCE_SIZE), mode='bilinear')
                        
                        out_ack = ack_siam(img0=X0,img1=X1, mode = 'guo_crop_speed', model_attacked = tran_tracker ,speed = True ,crop = [x0_patches,x1_patches])
                        #print(img_attack.size())                  guo_crop_speed
                        #save_torchimg(out_ack['adv_x_crop'],'trans')
                        #print(out_ack['adv_x_crop'].size(),tracker.img_sample_sz)
                        dimp_sz = tracker.img_sample_sz[0].int().item()
                        #print(dimp_sz)
                        resize_blur  = torch.nn.functional.interpolate(out_ack['adv_x_crop'], size=(dimp_sz,dimp_sz), mode='bilinear')
                        #print(resize_blur.size())
                        outputs = tracker.track(img,  x_crop  = resize_blur ,coord = pos ,ispert=True)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input() 
                    else: 
                        #X0 = X1.copy()
                        #print(img.shape)
                        X0 = img.copy()
                        outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    if box1 is not None :
                        box0 = box1
                    box1 = pred_bbox
                    #print(box0,box1)
                    #d_m = get_move_dis(box0,box1)
                    
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if  idx %5 == 0 and attack_bool and test_mode == 'visionall' and vis_img :
                    print(idx)
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    pred_bbox_normblur = list(map(int, pred_bbox_normblur))
                    pred_bbox_advblur  = list(map(int, pred_bbox_advblur))
                    
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 1)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (255, 255, 0), 1)
                    
                    cv2.rectangle(out_ack['normblur_org'], (pred_bbox_normblur[0], pred_bbox_normblur[1]),
                                  (pred_bbox_normblur[0]+pred_bbox_normblur[2], pred_bbox_normblur[1]+pred_bbox_normblur[3]), (255, 255, 0), 1)
                    
                    cv2.rectangle(out_ack['advblur_org'], (pred_bbox_advblur[0], pred_bbox_advblur[1]),
                                  (pred_bbox_advblur[0]+pred_bbox_advblur[2], pred_bbox_advblur[1]+pred_bbox_advblur[3]), (0, 255, 255), 1)
                    
                    
                    cv2.putText(img, str(idx)+'original', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    cv2.putText(out_ack['normblur_org'], 'normblur', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(out_ack['advblur_org'], 'advblur', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    vis.image(img.transpose(2,0,1))
                 
                    vis.image(out_ack['normblur_org'].transpose(2,0,1))
                    
                    vis.image(out_ack['advblur_org'].transpose(2,0,1))
                    vis_img = False
                    #cv2.imshow(video.name, img)
                    print('按一下')
                    input()
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
