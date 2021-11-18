'''
Reference:
[1] Towards Deep Learning Models Resistant to Adversarial Attacks
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
arXiv:1706.06083v3
'''
import torch
import numpy as np
import os
import sys
import cv2
import torch.nn as nn
from  visdom import Visdom

vis=Visdom(env="attacker")
import torch.nn.functional as F
import random
father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
if not father_dir in sys.path:
    sys.path.append(father_dir)
from attack.attack_base import AttackBase, clip_eta

def save_torchimg(img,name):
    save_im = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/cheng/pytracking-master/ltr/attack/{}.jpg'.format(name), save_im)



class IPGD(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 / 255.0, sigma = 6 / 255.0, nb_iter = 5,
                 norm = np.inf, DEVICE = torch.device('cuda:0'),
                 mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]), random_start = True):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.MSELoss()#.to(DEVICE)  # MSELoss   L1Loss
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.random_start = random_start

    def single_attack(self, net, inp, label, eta,dimp_filters, target = None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        num_sequences = inp.shape[0]
        adv_inp = inp + eta
       
        
        
        #vis.heatmap(label[0][0])
        #net.zero_grad()
        backbone_feat_cur_all = net.extract_backbone_features(adv_inp)
        backbone_feat_cur = backbone_feat_cur_all[net.classification_layer]
        backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])
        
        dimp_scores_cur = net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)     
        pred = dimp_scores_cur[:, :, :-1, :-1].contiguous()
        #vis.heatmap(pred[0][0])
        #pred = net(adv_inp)
        if target is not None:
            targets = torch.sum(pred[:, target])
            grad_sign = torch.autograd.grad(targets, adv_in, only_inputs=True, retain_graph = False)[0].sign()
            
        else:
            
            loss = self.criterion(pred, label)
            #print(loss.requires_grad)
            #loss.requires_grad = True
            #print(adv_inp.size())
            #print(loss)
            #print(adv_inp)
            #backbone_feat_cur.sum().backward()
            #grad_sign = adv_inp.grad#.sign()  
            #print(grad_sign)
            
            grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs=True, retain_graph = False)[0].sign()
            #print(grad_sign.size())
            
            
        #print(loss)
        adv_inp = adv_inp - 1*grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std +  self._mean
        
        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._std

        return eta

    def attack(self, net, inp, data ,label, dimp_filters,target = None):
        #save_torchimg(inp*255,'oriinput')
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()
        #print(label.size())
        #print(inp.mean())
        
        
        label_adv = torch.zeros_like(label)
        for r in range(label_adv.size(1)):

            xx = random.randint(1,16)
            yy = random.randint(1,16)
            label_adv[0,r,yy,xx] = 1
            label_adv[0,r,yy+1,xx] = 1
            label_adv[0,r,yy,xx+1] = 1
            label_adv[0,r,yy+1,xx+1] = 1
        #print(label_adv[0][0])
        inp.requires_grad = True
        eta.requires_grad = True

        #vis.heatmap(label_adv[0][0])
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label_adv, eta,dimp_filters, target)
            #print(i)
        #print(eta)
        
        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std
        #save_torchimg(adv_inp*255,'advinput')
        
        
        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)


class IPGD_after(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 / 255.0, sigma = 3/ 255.0, nb_iter = 5,
                 norm = np.inf, DEVICE = torch.device('cuda:0'),
                 mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]), random_start = True):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.MSELoss()#.to(DEVICE)  # MSELoss   L1Loss
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.random_start = random_start

    def single_attack(self, net, inp, label, eta,data,dimp_filters, target = None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        num_sequences = inp.shape[0]
        adv_inp = inp + eta
       
        
        
        #vis.heatmap(label[0][0])
        #net.zero_grad()
        backbone_feat_cur_all = net.extract_backbone_features(adv_inp)
        backbone_feat_cur = backbone_feat_cur_all[net.classification_layer]
        backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])
        
        dimp_scores_cur = net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)     
        dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

        predictor_input_data = {'input1': data['input1'], 'input2': backbone_feat_cur,
                                    'label_prev': data['label_prev'], 'anno_prev': data['anno_prev'],
                                    'dimp_score_prev': data['dimp_score_prev'], 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': data['state_prev'],
                                    'jitter_info': data['jitter_info']}

        predictor_output = net.predictor(predictor_input_data)

        pred = predictor_output['response']
        #print(pred.size())
        
        #vis.heatmap(pred[0][0])
        #pred = net(adv_inp)
        if target is not None:
            targets = torch.sum(pred[:, target])
            grad_sign = torch.autograd.grad(targets, adv_in, only_inputs=True, retain_graph = False)[0].sign()
            
        else:
            
            loss = self.criterion(pred, label)
            #print(loss.requires_grad)
            #loss.requires_grad = True
            #print(adv_inp.size())
            #print(loss)
            #print(adv_inp)
            #backbone_feat_cur.sum().backward()
            #grad_sign = adv_inp.grad#.sign()  
            #print(grad_sign)
            
            grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs=True, retain_graph = False)[0].sign()
            #print(grad_sign.size())
            
            
        #print(loss)
        adv_inp = adv_inp - 1*grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std +  self._mean
        
        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._std

        return eta

    def attack(self, net, inp, data,label, dimp_filters,target = None):
        #save_torchimg(inp*255,'oriinput')
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()
        #print(label.size())
        #print(inp.mean())
        
        
        label_adv = torch.zeros_like(label)
        for r in range(label_adv.size(1)):

            xx = random.randint(1,16)
            yy = random.randint(1,16)
            label_adv[0,r,yy,xx] = 1
            label_adv[0,r,yy+1,xx] = 1
            label_adv[0,r,yy,xx+1] = 1
            label_adv[0,r,yy+1,xx+1] = 1
        #print(label_adv[0][0])
        inp.requires_grad = True
        eta.requires_grad = True

        #vis.heatmap(label_adv[0][0])
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label_adv, eta,data, dimp_filters, target)
            #print(i)
        #print(eta)
        
        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std
        #save_torchimg(adv_inp*255,'advinput')
        
        
        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

class IPGD_siamkys(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 , sigma = 3 , nb_iter = 5,
                 norm = np.inf, DEVICE = torch.device('cuda:0'), random_start = True,
                 mean = 0 ,#torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = 1 ):#torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.MSELoss()#.to(DEVICE)  # MSELoss   L1Loss
        self.DEVICE = DEVICE
        self._mean = mean#.to(DEVICE)
        self._std = std#.to(DEVICE)
        self.random_start = random_start
        self.maxpool = nn.MaxPool2d(25)

    def single_attack(self, net, inp, label, eta, data ):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        num_sequences = inp.shape[0]
        adv_inp = inp + eta
       
        
        
        #vis.heatmap(label[0][0])
        #net.zero_grad()
        zf = data['zf']
        xf = net.backbone(adv_inp)
        xf = net.neck(xf)
        feat2 = net.adjcon1(xf[2])
        feat2 = net.adjcon2(feat2)
        cls, loc = net.rpn_head(zf, xf)
        bats = cls.size(0)
        score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].view(bats, 5,25,25)
        score_maxid= self.maxpool(score).view(bats,-1)
        #score_map = score.sum(1)/5
        #print(score_maxid.size())
        #print(score_maxid)
        _ , maxid = torch.max(score_maxid,dim=1)
        #print(maxid)
        score_ls=[]
        for i in range(len(maxid)):
            score_ls.append(score[i,maxid[i],:,:])
        score_one = torch.stack(score_ls)
        #print(score_one.size())
        #vis.heatmap(score_one[0])
        
       
        
        rnn_data={
            'feat1':data['xf_prev'],
            'feat2':feat2,
            'dimp_score_cur':score_one,
            'state_prev': data['state_prev'],
            'label_prev':data['label_prev']
        }


        #print(len(xf))
        
        output = net.RNN_defence(rnn_data)

        #print(label.size())
        
        pred = output['response'].unsqueeze(0) 
        #vis.heatmap(pred[0][0])
        #pred = net(adv_inp)
        
        if label is not None:
            loss = self.criterion(pred, label)
            #print(loss.requires_grad)
            #loss.requires_grad = True
            #print(adv_inp.size())
            #print(loss)
            #print(adv_inp)
            #backbone_feat_cur.sum().backward()
            #grad_sign = adv_inp.grad#.sign()  
            #print(grad_sign)
            
            grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs=True, retain_graph = False)[0].sign()
            #print(grad_sign.size())
            
            
        #print(loss)
        adv_inp = adv_inp - 1*grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std +  self._mean
        
        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 255) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._std

        return eta

    def attack(self, net, inp, data ):
        #save_torchimg(inp*255,'oriinput')
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()
        #print(label.size())
        #print(inp.mean())
        
        
        label_adv = torch.zeros_like(data['label_prev'])
        #print(label_adv.size())
        
        for r in range(label_adv.size(1)):

            xx = random.randint(1,22)
            yy = random.randint(1,22)
            label_adv[0,r,yy,xx] = 1
            label_adv[0,r,yy+1,xx] = 1
            label_adv[0,r,yy,xx+1] = 1
            label_adv[0,r,yy+1,xx+1] = 1
        #print(label_adv[0][0])
        inp.requires_grad = True
        eta.requires_grad = True

        #vis.heatmap(label_adv[0][0])
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label_adv, eta, data )
            #print(i)
        #print(eta)
        
        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 255)
        adv_inp = (tmp_adv_inp - self._mean) / self._std
        #save_torchimg(adv_inp*255,'advinput')
        
        
        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

class IPGD_siamrpn(AttackBase):
    # ImageNet pre-trained mean and std
    # _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])

    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 , sigma = 3 , nb_iter = 5,
                 norm = np.inf, DEVICE = torch.device('cuda:0'), random_start = True,
                 mean = 0 ,#torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = 1 ):#torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.MSELoss()#.to(DEVICE)  # MSELoss   L1Loss
        self.DEVICE = DEVICE
        self._mean = mean#.to(DEVICE)
        self._std = std#.to(DEVICE)
        self.random_start = random_start
        self.maxpool = nn.MaxPool2d(25)

    def single_attack(self, net, inp, label, eta, data ):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        num_sequences = inp.shape[0]
        adv_inp = inp + eta
       
        
        
        #vis.heatmap(label[0][0])
        #net.zero_grad()
        zf = data['zf']
        xf = net.backbone(adv_inp)
        xf = net.neck(xf)
        #feat2 = net.adjcon1(xf[2])
        #feat2 = net.adjcon2(feat2)
        cls, loc = net.rpn_head(zf, xf)
        bats = cls.size(0)
        score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        pred = F.softmax(score, dim=1)[:, 1].view(bats, 5,25,25).sum(1).unsqueeze(0)
        #print(pred)
        #print(score_one.size())
        #vis.heatmap(score_one[0])
        
        
        #vis.heatmap(pred[0][0])
        #pred = net(adv_inp)
        
        if label is not None:
            loss = self.criterion(pred, label)
            #print(loss.requires_grad)
            #loss.requires_grad = True
            #print(adv_inp.size())
            #print(loss)
            #print(adv_inp)
            #backbone_feat_cur.sum().backward()
            #grad_sign = adv_inp.grad#.sign()  
            #print(grad_sign)
            
            grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs=True, retain_graph = False)[0].sign()
            #print(grad_sign.size())
            
        
        #print(loss)
        adv_inp = adv_inp - 1*grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std +  self._mean
        
        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 255) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._std

        return eta

    def attack(self, net, inp, data ):
        #save_torchimg(inp*255,'oriinput')
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()
        #print(label.size())
        #print(inp.mean())
        
        
        label_adv = torch.zeros_like(data['label_prev'])
        #print(label_adv.size())
        
        for r in range(label_adv.size(1)):

            xx = random.randint(1,22)
            yy = random.randint(1,22)
            label_adv[0,r,yy,xx] = 1
            label_adv[0,r,yy+1,xx] = 1
            label_adv[0,r,yy,xx+1] = 1
            label_adv[0,r,yy+1,xx+1] = 1
        #print(label_adv[0][0])
        inp.requires_grad = True
        eta.requires_grad = True

        #vis.heatmap(label_adv[0][0])
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label_adv, eta, data )
            #print(i)
        #print(eta)
        
        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 255)
        adv_inp = (tmp_adv_inp - self._mean) / self._std
        #save_torchimg(adv_inp*255,'advinput')
        
        
        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

def test_IPGD():
    pass
if __name__ == '__main__':
    test_IPGD()
