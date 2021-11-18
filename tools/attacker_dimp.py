import time
import os
from torch.autograd import Variable
import math
import torch

import random
import numpy as np
import numpy
#from DAIN.networks.DAIN import DAIN
from pysot.core.config import cfg
#from DAIN.my_args import  args
from DAIN import PWCNet
#from scipy.misc import imread, imsave
from DAIN.AverageMeter import  *
import torch.nn as nn
from torch.nn import functional as F
from  visdom import Visdom
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torchvision
from pysot.models.loss import select_cross_entropy_loss
from pysot.models.backbone.resnet_atrous import ResNet,Bottleneck
from pysot.utils.model_load import load_pretrain
from extern.pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
import cv2
from pysot.utils.bbox import center2corner, Center, get_axis_aligned_bbox
from pysot.datasets.anchor_target import AnchorTarget
import warnings
from torch.autograd import Variable



warnings.filterwarnings("ignore")

vis=Visdom(env="attacker")
divnum = 15      
backbone_path = '/CHENGZIYI/pysot-blur/resnet50.model'
learning_rate = 10000
tv_beta = 2
l1_coeff= 1 
tv_coeff= 0.05
abs_coeff=0.00005
max_iterations = 10
tnum = 10
epsilon =   0.0002    #0.0002  白盒全图
epsilon_s = 0.002
flow_scale = 20.0
xl_lr = 1
yl_lr = 1
vis_img = True
#a_x = 3  # 正为加速

lamb_mome = 0.00001
x_dev = 0
y_dev = 0
delta_scale = 0.5
output_org = False
singel =  False
mask_obj = False
MIFGSM = True
grad_norm = True
#speed = False
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def tv_norm(input, tv_beta):
                img = input
                #print(img.size())
                
                #print(img[1:, :].size())
                row_grad = torch.mean(torch.abs((img[:,:-1, :,:] - img[:,1:, :,:])).pow(tv_beta))
                col_grad = torch.mean(torch.abs((img[:,:, :-1,:] - img[:,:, 1:,:])).pow(tv_beta))
                return row_grad + col_grad
def save_torchimg(img,name):
    save_im = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/workspace/ABA/{}.jpg'.format(name), save_im)

class attacker(nn.Module):
    def __init__(self):
        super(attacker, self ).__init__() 
        #self.inter_model = DAIN(channel=3,filter_size = 4,timestep=0.5,training=False).eval().cuda()
    
        self.flownets = PWCNet.__dict__['pwc_dc_net']("DAIN/PWCNet/pwc_net.pth.tar").cuda()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], [2,3,4]).cuda()
        #load_pretrain(self.backbone, backbone_path)
        self.prev_delta = None
        self.inta = 10

    def img_to_torch(self,img):
            return torch.from_numpy( np.transpose(img, (2,0,1)).astype("float32")/ 255.0).type(torch.cuda.FloatTensor)
    
    def get_flow(self,X0,X1,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth):
            
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            #print(sam0.size())
            #print(X0.size())
            
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1)
            #print(cur_offset_input.size())
            
            flow = self.flownets(cur_offset_input)
            temp = flow *flow_scale
            #temp.clone().requires_grad = True
            #x = Variable(torch.rand(3,3,25,25), requires_grad=True)
            #y = nn.Upsample(size=(17,17) ,mode='bilinear')(x)
            #y.sum().backward()
           
            flow_full = nn.Upsample(scale_factor=4, mode='bilinear')(temp)
            
        
            #flow_full = flip(flow_full,2)   #
            
            #vis.heatmap(flow_full[0][0])
            #vis.heatmap(flow_full[0][1]) 
            #vis.heatmap(flow_new[0][0])
            
            #vis.heatmap(flow_new[0][1])
            #print(flow_full.size())
            #print(img1.shape)
            
            #vis.image(flow_full[:,0,:,:])
            #vis.image(flow_full[:,1,:,:])
            flow_full = flow_full[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(0,2,3,1)#
            
            return  flow_full
        
    def attack_once(self, tracker, data):
       
        search = data['search'].cuda()
        alpha_samll = data['alpha_samll'].cuda()
        alpha_org = data['alpha_org'].cuda()
        adv_cls = data['adv_cls'].cuda()
        momen = data['momen'].cuda()
        #prev_perts = data['prev_perts'].cuda()   'alpha_samll':alpha_samll
        #print(search.size(), alpha_samll.size())
        search = (search *alpha_samll).sum(0).unsqueeze(0) 
        #print(search,search.size())
        
        
      
        backbone_feat = tracker.net.extract_backbone(search)
        #print(backbone_feat)                                                          
        test_x = tracker.net.extract_classification_feat(backbone_feat)
        scores = tracker.net.classifier.classify(tracker.target_filter,test_x)
            
        #vis.heatmap(scores[0][0])
        cls_loss = torch.norm((scores-adv_cls),p=2)
        
        
        
        
        
     
        reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
        total_loss = cls_loss + 0.0005 * reg_loss
        #total_loss = test_x.sum()       #backbone_feat['layer2'].sum()  
        
        #print(total_loss)
        total_loss.backward()
       


        x_grad = -data['alpha_samll'].grad   
        #print(x_grad)   
        
        #
        if MIFGSM:
            momen = lamb_mome*momen+x_grad/torch.norm(x_grad,1)
            alpha_samll = alpha_samll + epsilon *torch.sign(momen)
        else:
            
            x_grad = torch.sign(x_grad)
            #print(x_grad.sum())
            alpha_samll = alpha_samll + epsilon * x_grad
        
        #print(alpha_samll.size(),alpha_samll[:,0,55,55])
        
        #alpha_samll = F.softmax(alpha_samll,dim=0)
        #print(alpha_samll.size(),alpha_samll[:,0,55,55])
        
        #input()
        '''
        pert = adv_x - search-pert_sum
        norm = torch.sum(torch.abs(pert))
        pert = torch.min(pert*self.inta/norm, pert)

        p_search = search + pert + pert_sum
        p_search = torch.clamp(p_search, 0, 255)
        pert = p_search - search - prev_perts.sum(0)
        '''
        return alpha_samll, total_loss
    
    def flow_subwindow(self,im, pos, model_sz, original_sz, avg_chans):
        
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.float32)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = 0
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = 0
            if left_pad:
                te_im[:, 0:left_pad, :] = 0
            if right_pad:
                te_im[:, c + left_pad:, :] = 0
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        #print(im_patch.shape)
    
        #vis.heatmap(im_patch[0][0])
        
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch
    
    def forward(self,img0,img1,mode ='guo',model_attacked = None,tracker =None,speed =True,crop = None):
        if mode == 'whitebox' or mode =='nor_blur':
            
            
            tracker = model_attacked
            if imlist is not None :
    
                X1 = imlist[-1]
                X0 = imlist[-2]
            X0 = self.img_to_torch(X0)
            X1 = self.img_to_torch(X1)
            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))
            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            #print(sam0.size())
            #print(X0.size())
            
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1)
            flow = self.flownets(cur_offset_input)
            temp = flow * flow_scale#*0.5
            flow_full = nn.Upsample(scale_factor=4, mode='bilinear')(temp)
            #print(flow_full.size())
            #print(X0.shape)
            
            #vis.image(flow_full[:,0,:,:])
            #vis.image(flow_full[:,1,:,:])
            flow_full = flow_full.permute(0,2,3,1)
            sum_img01 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            sum_img10 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            xl,yl = torch.meshgrid(torch.Tensor(range(X0.size(2))),torch.Tensor(range(X0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            theta01 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            theta10 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            
            
            #print(idflow.size())
            #vis.image(X0.squeeze(0))
            #vis.image(X1.squeeze(0))
            #print(X0)
            #flow_norm = torch.zeros_like(flow_full )
            #print(alpha)
            
            
            
            for i in range(divnum):
                theta = (i) * flow_full / divnum
                #print(theta.size())
                #print(idflow.size())
                    
                #print(theta01.size())
                #print((idflow - theta).size())
                theta01[i]  = (idflow - theta).squeeze(0)
                theta10[i]  = (idflow + theta).squeeze(0)
                    
            theta01[:,:,:,0]=(torch.clamp(theta01[:,:,:,0],0,X0.size(3))-X0.size(3)/2)/(X0.size(3)/2)
            theta01[:,:,:,1]=(torch.clamp(theta01[:,:,:,1],0,X0.size(2))-X0.size(2)/2)/(X0.size(2)/2)
                        
            theta10[:,:,:,0]=(torch.clamp(theta10[:,:,:,0],0,X0.size(3))-X0.size(3)/2)/(X0.size(3)/2)
            theta10[:,:,:,1]=(torch.clamp(theta10[:,:,:,1],0,X0.size(2))-X0.size(2)/2)/(X0.size(2)/2)
            x0input = X0.repeat(divnum,1,1,1)
            x1input = X1.repeat(divnum,1,1,1)
            #重采样
            warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
            warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
            out_img = torch.cat((warped01,warped10),0)

            
            
            #print(X1.size())      # [0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
            out_img =  out_img.clamp(0,1.0)[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(0,2,3,1)
            #print(out_img.size())
            
            label_in =  X1.clamp(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(1,2,0)
            #定义 融合参数
            alpha_org = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            alpha = torch.ones(divnum*2,out_img.size(1),out_img.size(2),1).cuda()      #  randn    +1
            #print(alpha.size())
            
            tra_im = out_img.cpu().detach().numpy()*255
            mix_crop = torch.zeros(2*divnum,3,255,255)

            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            #构造切片合集
            for i in range(2*divnum):
                img_mid = tracker.get_subwindow(tra_im[i], tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
                mix_crop[i]=img_mid[0]

            if mode=='nor_blur':
                #nor_w = torch.ones(2*divnum,1,out_img.size(1),out_img.size(2)).cuda() 
                out_img = (mix_crop.cuda()  * alpha_org).sum(0).unsqueeze(0)
                #print(out_img.size())
                return out_img
     
            label_in = tracker.get_subwindow(label_in.cpu().numpy(), tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            #alpha = alpha.requires_grad_()
            
            
            clean_cls , clean_loc = model_attacked.attack_track(label_in,x_cr=label_in)
            #cls_out = model_attacked.model.log_softmax(cls_p)
            #print(cls_out.size())
            
            #print(clean_out[:1056].size())
            v , ida = torch.topk(clean_cls,k= tnum,largest=False,sorted = False)
            vb, ida_big = torch.topk(clean_cls,k= tnum,largest=True,sorted = False)
            #print(ida)
            #vis.bar(clean_cls)
            #vis.bar(clean_loc[0])
            #inde = torch.tensor(ida)

            ida =[1555]
            label_cls = (torch.max(torch.abs(clean_cls)) - torch.abs(clean_cls)).detach()
           
            alpha_samll = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)#.requires_grad_()     pert + pert_sum.detach()
            
            pert_sum = torch.zeros(2*divnum,3,255,255).cuda()
            #optimizer = torch.optim.SGD([alpha_samll], lr=learning_rate,momentum=0.3)  #  momentum=0.9
            or_crop  = mix_crop
            for _ in range(max_iterations):
                alpha_samll = alpha_samll.requires_grad_()
                #pert = torch.zeros(2*divnum,1,255,255).cuda().requires_grad_()
                img_mix = (mix_crop.detach().cuda() * alpha_samll).sum(0).unsqueeze(0)
                #print(img_mix)
                 
                #pert = pert.requires_grad_()
                #out_img = Variable(out_img , requires_grad=True)
                #img_mix = (out_img.detach() + pert).sum(0) /(2*divnum)
                #print(img_mix.size())
                #vis.image(img_mix.squeeze(0))
                track_in = img_mix.clamp(0,1.0)    #[:, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
                #print(track_in.size())
               
                attack_cls, attack_loc = tracker.attack_track(track_in, x_cr=track_in)
                #print(attack_out.size())
                
                x_l =0.0
                y_l =0.0
                for i in ida:
                    #print(q)
                    x_l +=torch.abs(attack_cls[i])
                    #y_l +=torch.abs(attack_loc[1][i])
                    #print(attack_out[q])
                for i in ida_big:
                    #print(q)
                    y_l +=torch.abs(attack_cls[i])
                    #y_l +=torch.abs(attack_loc[1][i])
                    #print(attack_out[q])
                #aa = attack_out.index_select(1,inde)
                #print(x_l)
                #print(y_l)
                
                #print(tt.requires_grad)
                #vis.bar(clean_out)
                #vis.bar(label)
                #vis.bar(attack_out)
                
                #print(attack_out.requires_grad)
                loss1 = l1_coeff  *  (torch.abs(attack_cls-label_cls)).sum() #+ l1_coeff  *  torch.mean(torch.abs(attack_loc[1]-label_1))
                print(loss1.sum())
                #loss1 = -x_l  # y_l  
                loss2 = tv_coeff * tv_norm(alpha, tv_beta)
                loss3 = abs_coeff * torch.norm(alpha_samll-alpha_org, 2)   #torch.mean(torch.abs(pert))
                loss =  loss1   + loss3
                #optimizer.zero_grad()
                #print(alpha.requires_grad)
                loss.sum().backward()
                #print(alpha)
                #print(alpha.grad)
                signs = torch.sign(alpha_samll.grad)
                
                #print(loss1,loss2,loss3)
                alpha_samll = (alpha_samll - epsilon * signs).detach()#.requires_grad_()
                #alpha_samll = (alpha_samll - 10000000000 * alpha_samll.grad).detach()
                #print(signs.sum())
                #pert_sum = (pert_sum + pert).detach()
                #pert =  alpha_samll-alpha_org
                #mix_crop = mix_crop.cuda() + alpha_samll
                #print(alpha)
                #optimizer.step()
                #print('################################')
                #input()
            '''
            
           
            for q in ida:
                l +=attack_out[q]
                print(attack_out[q])
                #aa = attack_out.index_select(1,inde)
            print(l)
            '''
            vb, ida_out = torch.topk(attack_cls,k= tnum,largest=True,sorted = False)
            #print(ida_big)
            #print(ida_out)
            img_mix = (or_crop.cuda() * (alpha_samll + pert_sum)).sum(0)
            #img_blur = (out_img * alpha).sum(0) /(2*divnum)   #.unsqueeze(0)
            #img_blur = img_blur.clamp(0,1.0)*255
            #img_blur = track_in*255.0
            #print(img_blur.size())
            
            #vis.bar(clean_out)
            
            
            #print(attack_out)
            #vis.bar(clean_cls)
            #input()
            #vis.bar(attack_cls)
            #vis.image((alpha-alpha_org)[0].permute(2,0,1))  .permute(2,0,1)
            #vis.image(img_mix)
            #
            
            #vis.image((out_img.sum(0) /(2*divnum)).permute(2,0,1).clamp(0,1.0)*255)
            return  img_mix.unsqueeze(0)      #img_blur.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        elif mode =='guo_speed':
            tracker = model_attacked
               
            X0 = self.img_to_torch(img0)
            X1 = self.img_to_torch(img1)
            #print(X1.size())
            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))
            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            #print(sam0.size())
            #print(X0.size())
            
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1)
            #print(cur_offset_input.size())
            
            flow = self.flownets(cur_offset_input)
            temp = flow *flow_scale
            
            flow_full = nn.Upsample(scale_factor=4, mode='bilinear')(temp)
            
            #flow_vis = flip(flow_full,2)   # vis.heatmap 上下方向不对，所以需要翻转过来看
            
            #vis.heatmap(flow_vis[0][0])
            #vis.heatmap(flow_vis[0][1]) 
            #vis.heatmap(flow_new[0][0])
            
            #vis.heatmap(flow_new[0][1])
            #print(flow_full.size())
            ##print(img1.shape)
            
            #vis.image(flow_full[:,0,:,:])
            #vis.image(flow_full[:,1,:,:])
            full_flow = flow_full[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(0,2,3,1)  
            #flow_full[:,:,:,1]=  -flow_full[:,:,:,1]
            sum_img01 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            sum_img10 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            
            theta01 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            theta10 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            
            
            #print(full_flow.size())
            
            flow_x = full_flow[0,:,:,0].repeat(1,3,1,1)#.permute(1,2,0)#.detach()#.cpu().numpy()
            flow_y = full_flow[0,:,:,1].repeat(1,3,1,1)#.permute(1,2,0)#.detach()#.cpu().numpy()
            #print(flow_y.size())
            flow_x, _  = sample_patch_multiscale(flow_x, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))
            flow_y, _  = sample_patch_multiscale(flow_y, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))
            #print(flow_y.size())     
            flow_full_crop = torch.stack((flow_x[0][0],flow_y[0][0])).permute(1,2,0).unsqueeze(0)   
            #print(flow_full_crop.size())
            
            
            
            
            #vis.image((normblur_org.permute(2,0,1)))
        
            label_in =  X1[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255     #.permute(1,2,0)
            x0_in =  X0[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255 
            frame_0, _ = sample_patch_multiscale(label_in, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
            
            #print(out_img.size(),label_in.size()) # torch.Size([50, 432, 576, 3])
            #生成GT 
            
            x1_patches, patch_coords = sample_patch_multiscale(label_in, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))  
            backbone_feat = tracker.net.extract_backbone(x1_patches)                                                                                                                  
            #print(label_in)                                                          
            test_x = tracker.get_classification_features(backbone_feat)
            scores_gt = tracker.classify_target(test_x)  
            frame_1 = x1_patches

            #print(scores_gt.size()) 
            #vis.image(label_in[0])
            #vis.heatmap(scores_gt[0][0])
            adv_cls = self.get_adv_lab(scores_gt).cuda() 

            #vis.heatmap(adv_cls[0][0])
            

            
            
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            #print(mix_patches.size())
            
            x0input = frame_0.repeat(divnum,1,1,1).detach()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach()
      
            alpha_samll = torch.ones(2*divnum,1,224, 224).cuda()/(2*divnum)
            alpha_org = torch.ones(2*divnum,1,224, 224).cuda()/(2*divnum)
            sx = torch.rand((divnum)).cuda()#.requires_grad = True
            sy = torch.rand((divnum)).cuda()
            sx = F.softmax(sx).detach()
            sy = F.softmax(sy).detach()
            momen = torch.zeros(frame_1.size()).cuda()
            losses=[]
            m = 0
            firs =True
            #print(x_crop)
            while m < max_iterations:
                
                theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                flow_full_crop = flow_full_crop.detach()
                cur_x = frame_1.clone().detach()#/255.0
                idflow = idflow.detach()
                frame_0 = frame_0.detach()
                frame_1 = frame_1.detach()
                sx = sx / sx.sum(0).unsqueeze(0)
                sy = sy / sy.sum(0).unsqueeze(0)
                #sx = F.softmax(sx).detach()
                #sy = F.softmax(sy).detach()
                sx.requires_grad = True
                sy.requires_grad = True
                momen = momen.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
               
                #print(sx,sy)
                
                for i in range(divnum):
                    if speed :
                        theta_fw_x =    (sx[:i+1].sum())     * flow_full_crop[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        theta_fw_y =    (sy[:i+1].sum())     * flow_full_crop[:,:,:,1]
                        theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                    else:
                        theta_fw = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                      
                   
                    #input()
                    theta_bw = theta_fw - flow_full_crop 
                    #print(idflow.size(),theta_fw.size())     torch.clamp    ,0,frame_0.size(3)
                    theta01[i]  = (idflow - theta_fw).squeeze(0)
                    theta10[i]  = (idflow - theta_bw).squeeze(0)
                
                theta01[:,:,:,0]=((theta01[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta01[:,:,:,1]=((theta01[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #vis.heatmap(theta01[15,:,:,0])  
                #vis.heatmap(theta01[15,:,:,1])         
                theta10[:,:,:,0]=((theta10[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta10[:,:,:,1]=((theta10[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #theta01.sum().backward() 
                #theta01.sum().backward() 
           
                
                #vis.heatmap(theta01[0,:,:,0])
                warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
           
                #vis.image(warped10[0])
                
                
                mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
                #print(mix_crop.size())
                
                blur_search = (mix_crop * alpha_samll).sum(0).unsqueeze(0)  # 整个search 模糊， 要mask出object
                #vis.heatmap(cur_x[0][1])
                #print(blur_search.size())
                #vis.image(blur_search[0])
                
                #vis.heatmap(blur_search[0][2])
               
                while blur_search.max() >255:
                    blur_search = blur_search * 0.95
                if mask_obj:
                    cur_x[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] = blur_search[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] 
                else:
                    cur_x = blur_search
               
                #print(cur_x.size())
                
      
                backbone_feat = tracker.net.extract_backbone(cur_x)
                #print(backbone_feat)                                                          
                test_x = tracker.net.extract_classification_feat(backbone_feat)
                scores = tracker.net.classifier.classify(tracker.target_filter,test_x)
                    
                #vis.heatmap(scores[0][0])
                cls_loss = torch.norm((scores-adv_cls),p=2)
        
                reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
                total_loss = cls_loss + 0.0005 * reg_loss
                #total_loss = test_x.sum()       #backbone_feat['layer2'].sum()  
                
                #print(total_loss)
                total_loss.backward()
            


                 
                #print(sx,sy)
                w_grad = -alpha_samll.grad   
                if speed :
                    sx_grad = -sx.grad  
                    sy_grad = -sy.grad
                #
                if MIFGSM:
                    momen = lamb_mome*momen+w_grad/torch.norm(w_grad,1)
                    alpha_samll = (alpha_samll + epsilon *torch.sign(momen)).detach()
                    if speed :
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        #print(sx,sy)
                else:
                    
                    w_grad = torch.sign(w_grad)
                    #print(x_grad.sum())
                    alpha_samll = (alpha_samll + epsilon * w_grad).detach()
                    if speed and firs:
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        sx = F.softmax(sx).detach()
                        sy = F.softmax(sy).detach()
                        firs =False
                    
                #print(sx,sy)
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                #vis.image(cur_x[0])
                m = m+1
            #print(losses)
           
            
            if output_org:
                    advblur_org = tracker.get_orgimg(img1, adv_x_crop.unsqueeze(0), tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
            
            out={
                #'advblur_org':advblur_org,
                #'normblur_org':(normblur_org.cpu().detach().numpy())*255,
                'adv_x_crop':cur_x,
                'coord': patch_coords
                #'x_crop_normblur':x_crop_normblur.permute(2,0,1).unsqueeze(0)
            }
            return out
            
        elif mode == 'guo_fine_speed':
            tracker = model_attacked
               
            X0 = self.img_to_torch(img0)
            X1 = self.img_to_torch(img1)
            #print(X1.size())
            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))
            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            #print(sam0.size())
            #print(X0.size())
            
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1)
            #print(cur_offset_input.size())
            
            flow = self.flownets(cur_offset_input)
            temp = flow *flow_scale
            
            flow_full = nn.Upsample(scale_factor=4, mode='bilinear')(temp)
            
            #flow_vis = flip(flow_full,2)   # vis.heatmap 上下方向不对，所以需要翻转过来看
            
            #vis.heatmap(flow_vis[0][0])
            #vis.heatmap(flow_vis[0][1]) 
            #vis.heatmap(flow_new[0][0])
            
            #vis.heatmap(flow_new[0][1])
            #print(flow_full.size())
            ##print(img1.shape)
            
            #vis.image(flow_full[:,0,:,:])
            #vis.image(flow_full[:,1,:,:])
            full_flow = flow_full[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(0,2,3,1)  
            #flow_full[:,:,:,1]=  -flow_full[:,:,:,1]
            sum_img01 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            sum_img10 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            
            theta01 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            theta10 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            
            
            #print(full_flow.size())
            
            flow_x = full_flow[0,:,:,0].repeat(1,3,1,1)#.permute(1,2,0)#.detach()#.cpu().numpy()
            flow_y = full_flow[0,:,:,1].repeat(1,3,1,1)#.permute(1,2,0)#.detach()#.cpu().numpy()
            #print(flow_y.size())
            flow_x, _  = sample_patch_multiscale(flow_x, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))
            flow_y, _  = sample_patch_multiscale(flow_y, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))
            #print(flow_y.size())     
            flow_full_crop = torch.stack((flow_x[0][0],flow_y[0][0])).permute(1,2,0).unsqueeze(0)   
            #print(flow_full_crop.size())
            
            
            
            
            #vis.image((normblur_org.permute(2,0,1)))
        
            label_in =  X1[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255     #.permute(1,2,0)
            x0_in =  X0[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255 
            frame_0, _ = sample_patch_multiscale(x0_in, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate')) 
            
            
            #print(out_img.size(),label_in.size()) # torch.Size([50, 432, 576, 3])
            #生成GT 
            
            frame_1, patch_coords = sample_patch_multiscale(label_in, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))  
            if crop is not None :
                frame_0 = crop[0]
                frame_1 = crop[1]
            backbone_feat = tracker.net.extract_backbone(frame_1)                                                                                                                  
            #print(label_in)                                                          
            test_x = tracker.get_classification_features(backbone_feat)
            scores_gt = tracker.classify_target(test_x)  
            

            #print(scores_gt.size()) 
            
            #vis.image(label_in[0])
            #vis.heatmap(scores_gt[0][0])
            adv_cls = self.get_adv_lab(scores_gt).cuda() 

            #vis.heatmap(adv_cls[0][0])
            
            img_sz = frame_0.size(2)
            
            
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
           
            x0input = frame_0.repeat(divnum,1,1,1).detach().cuda()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach().cuda()
      
            alpha_samll = torch.ones(2*divnum,1,img_sz, img_sz).cuda()/(2*divnum)
            alpha_org = torch.ones(2*divnum,1,img_sz, img_sz).cuda()/(2*divnum)
            sx = torch.rand((divnum,1,img_sz,img_sz)).cuda()#.requires_grad = True
            sy = torch.rand((divnum,1,img_sz,img_sz)).cuda()
            #sx = torch.rand((divnum)).cuda()#.requires_grad = True
            #sy = torch.rand((divnum)).cuda()
            sx = F.softmax(sx,dim=0).detach()
            sy = F.softmax(sy,dim=0).detach()
            momen = torch.zeros(frame_1.size()).cuda()
            losses=[]
            m = 0
            firs =True
            last = False
            #print(x_crop)
            while m < max_iterations:
                
                theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                flow_full_crop = flow_full_crop.detach()
                cur_x = frame_1.clone().detach()#/255.0
                idflow = idflow.detach()
                frame_0 = frame_0.detach()
                frame_1 = frame_1.detach()
                sx = sx / sx.sum(0).unsqueeze(0)
                sy = sy / sy.sum(0).unsqueeze(0)
                #print(sx.size(),sx[:,:,3,3])
                if speed :
                    sx.requires_grad = True
                    sy.requires_grad = True
                momen = momen.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
               
                #print(sx,sy)
                
                for i in range(divnum):
                    if speed :
                        theta_fw_x =    (sx[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        theta_fw_y =    (sy[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,1]
                        #theta_fw_x =    (sx[:i+1].sum(0))     * flow_full_crop[:,:,:,0]
                        #theta_fw_y =    (sy[:i+1].sum(0))     * flow_full_crop[:,:,:,1]
                        theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                    else:
                        theta_fw = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                      
                   
                    #input()
                    theta_bw = theta_fw - flow_full_crop 
                    #print(idflow.size(),theta_fw.size())     torch.clamp    ,0,frame_0.size(3)
                    theta01[i]  = (idflow - theta_fw).squeeze(0)
                    theta10[i]  = (idflow - theta_bw).squeeze(0)
                
                theta01[:,:,:,0]=((theta01[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta01[:,:,:,1]=((theta01[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #vis.heatmap(theta01[15,:,:,0])  
                #vis.heatmap(theta01[15,:,:,1])         
                theta10[:,:,:,0]=((theta10[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta10[:,:,:,1]=((theta10[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #theta01.sum().backward() 
                #theta01.sum().backward() 
           
                
                #vis.heatmap(theta01[0,:,:,0])
                warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
           
                #vis.image(warped10[0])
                
                
                mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
                #print(mix_crop.size())
                
                blur_search = (mix_crop * alpha_samll).sum(0).unsqueeze(0)  # 整个search 模糊， 要mask出object
                #vis.heatmap(cur_x[0][1])
                #print(blur_search.size())
                #vis.image(blur_search[0])
                
                #vis.heatmap(blur_search[0][2])
               
                while blur_search.max() >255:
                    blur_search = blur_search * 0.95
                if mask_obj:
                    cur_x[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] = blur_search[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] 
                else:
                    cur_x = blur_search
               
                #print(cur_x.size())
                
      
                backbone_feat = tracker.net.extract_backbone(cur_x)
                #print(backbone_feat)                                                          
                test_x = tracker.net.extract_classification_feat(backbone_feat)
                scores = tracker.net.classifier.classify(tracker.target_filter,test_x)
                    
                #vis.heatmap(scores[0][0])
                #spx_loss = torch.std(sx)
                #spy_loss = torch.std(sy)
                
                
                cls_loss = torch.norm((scores-adv_cls),p=2)
        
                reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
                total_loss = cls_loss + 0.0005 * reg_loss #+ 0 * (spx_loss+ spy_loss)
                #total_loss = test_x.sum()       #backbone_feat['layer2'].sum()  
                
                
                total_loss.backward()
            


                 
                #print(sx[:,:,3,5])
                #print(sx,sy)
                w_grad = -alpha_samll.grad   
                if speed :
                    sx_grad = -sx.grad  
                    sy_grad = -sy.grad
                #
                if MIFGSM:
                    momen = lamb_mome*momen+w_grad/torch.norm(w_grad,1)
                    alpha_samll = (alpha_samll + epsilon *torch.sign(momen)).detach()
                    if speed :
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        #print(sx,sy)
                else:
                    
                    w_grad = torch.sign(w_grad)
                    #w_grad = 0
                    #print(x_grad.sum())
                    alpha_samll = (alpha_samll + epsilon * w_grad).detach()
                    if speed :
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        #sx = F.softmax(sx).detach()
                        #sy = F.softmax(sy).detach()
                        #firs =False
                
                if grad_norm:
                    syg_sum = torch.abs(sy_grad.view(divnum,-1)).sum(1)
                    sxg_sum = torch.abs(sx_grad.view(divnum,-1)).sum(1)
                    sx_min ,xid = torch.topk(sxg_sum,1,largest=False)
                    sy_min ,yid = torch.topk(syg_sum,1,largest=False)
                    #print(xid,yid)

                    res_x = sx.sum(0) - sx[xid]
                    res_y = sy.sum(0) - sy[xid]
                    #print(res_x,res_y)
                    sx[xid] = 1 - res_x
                    sy[xid] = 1 - res_y
                #print(total_loss)
                #print(sx.sum(0))
                
                #print(sx,sy)
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                
                m = m+1
            
            #vis.image(cur_x[0])
            #save_torchimg(cur_x,'dimp')
            #input()
            if output_org:
                    advblur_org = tracker.get_orgimg(img1, adv_x_crop.unsqueeze(0), tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
            #input()
            out={
                #'advblur_org':advblur_org,
                #'normblur_org':(normblur_org.cpu().detach().numpy())*255,
                'adv_x_crop':cur_x,
                'coord': patch_coords
                #'x_crop_normblur':x_crop_normblur.permute(2,0,1).unsqueeze(0)
            }
            return out
        elif mode =='guo':
            tracker = model_attacked
               
            X0 = self.img_to_torch(img0)
            X1 = self.img_to_torch(img1)
            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))
            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            #print(sam0.size())
            #print(X0.size())
            
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1)
            #print(cur_offset_input.size())
            
            flow = self.flownets(cur_offset_input)
            temp = flow *flow_scale
            
            flow_full = nn.Upsample(scale_factor=4, mode='bilinear')(temp)
            
            #flow_vis = flip(flow_full,2)   # vis.heatmap 上下方向不对，所以需要翻转过来看
            
            #vis.heatmap(flow_vis[0][0])
            #vis.heatmap(flow_vis[0][1]) 
            #vis.heatmap(flow_new[0][0])
            
            #vis.heatmap(flow_new[0][1])
            #print(flow_full.size())
            ##print(img1.shape)
            
            #vis.image(flow_full[:,0,:,:])
            #vis.image(flow_full[:,1,:,:])
            flow_full = flow_full.permute(0,2,3,1)
            #flow_full[:,:,:,1]=  -flow_full[:,:,:,1]
            sum_img01 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            sum_img10 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            xl,yl = torch.meshgrid(torch.Tensor(range(X0.size(2))),torch.Tensor(range(X0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            theta01 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            theta10 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            
            #print(idflow.size())
            #idflow[0,:,:,1] = flip(idflow[0,:,:,1],0)   #光流绝对坐标翻转， 左上角为0,0
            #vis.heatmap(idflow[0,:,:,0])
            #vis.heatmap(idflow[0,:,:,1])
            #print(idflow[0,:,:,1])
        
            #vis.image(X0.squeeze(0)[ :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth])
            #vis.image(X1.squeeze(0)[ :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth])
            #vis.heatmap(X0[0][0])
            #vis.heatmap(X1[0][0])
            #print(X0)
            #flow_norm = torch.zeros_like(flow_full )
            #print(alpha)
           
            #vis.heatmap(flow_full[0,:,:,0])
            #a_y = 2*(flow_full[:,:,:,1]-v0_y*divnum*unit_t)/(divnum*unit_t*divnum*unit_t)  # 计算约束的加速度
            #a_x = 2*(flow_full[:,:,:,0]-v0_x*divnum*unit_t)/(divnum*unit_t*divnum*unit_t)  # 计算约束的加速度
            #print(flow_full[0,213,494,1])
            for i in range(divnum):
                theta_f = (i+1) * flow_full / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                theta_b = (divnum-i-1) * flow_full / divnum
              
                #vis.heatmap((v0_y * cur_t + 0.5* a_y *cur_t*cur_t)[0,:,:])
                #vis.heatmap(theta_f[0,:,:,1])
                #print(theta_f[0,213,494,0])
                #input()
                theta_b = theta_f - flow_full
                theta01[i]  = (idflow - theta_f).squeeze(0)
                theta10[i]  = (idflow - theta_b).squeeze(0)
                    
            theta01[:,:,:,0]=(torch.clamp(theta01[:,:,:,0],0,X0.size(3))-X0.size(3)/2)/(X0.size(3)/2)
            theta01[:,:,:,1]=(torch.clamp(theta01[:,:,:,1],0,X0.size(2))-X0.size(2)/2)/(X0.size(2)/2)
            #vis.heatmap(theta01[15,:,:,0])  
            #vis.heatmap(theta01[15,:,:,1])         
            theta10[:,:,:,0]=(torch.clamp(theta10[:,:,:,0],0,X0.size(3))-X0.size(3)/2)/(X0.size(3)/2)
            theta10[:,:,:,1]=(torch.clamp(theta10[:,:,:,1],0,X0.size(2))-X0.size(2)/2)/(X0.size(2)/2)
            x0input = X0.repeat(divnum,1,1,1)
            x1input = X1.repeat(divnum,1,1,1)
            #重采样
            warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
            #print(warped01.size())
            if singel:
                warped01 =  warped01.clamp(0,1.0)[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
                warped01 = warped01.sum(0)/(divnum)
                vis.image(warped01)
                
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
                warped10 =  warped10.clamp(0,1.0)[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
                warped10 =  warped10.sum(0)/(divnum)
                vis.image(warped10)
                
                
            warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
            out_img = torch.cat((warped01,warped10),0)

            
            
            #print(X1.size())      # [0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
            out_img =  out_img.clamp(0,1.0)[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]*255    #.permute(0,2,3,1)
            #print(out_img.size())
            
            normblur_org = out_img.sum(0)/(2*divnum)
            #vis.image((normblur_org.permute(2,0,1)))
        
            label_in =  X1[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255     #.permute(1,2,0)
            
            
            alpha = torch.ones(divnum*2,out_img.size(1),out_img.size(2),1).cuda()      #  randn    +1
            #print(out_img.size(),label_in.size()) # torch.Size([50, 432, 576, 3])
            #生成GT 
            backbone_feat, _  = tracker.extract_backbone_features(label_in, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz)
            #print(label_in)                                                          
            test_x = tracker.get_classification_features(backbone_feat)
            scores_gt = tracker.classify_target(test_x)  

            #print(scores_gt.size()) 
            #vis.image(label_in[0])
            #vis.heatmap(scores_gt[0][0])
            adv_cls = self.get_adv_lab(scores_gt) 

            #vis.heatmap(adv_cls[0][0])
            

            
            
            mix_patches, patch_coords = sample_patch_multiscale(out_img, tracker.get_centered_sample_pos(),
                                                                      tracker.target_scale * tracker.params.scale_factors,
                                                                      tracker.img_sample_sz, getattr(tracker.params, 'border_mode', 'replicate'))
            
            #print(mix_patches.size())
            
            
      
            alpha_samll = torch.ones(2*divnum,1,224, 224).cuda()/(2*divnum)
            alpha_org = torch.ones(2*divnum,1,224, 224).cuda()/(2*divnum)
         
            momen = torch.zeros(mix_patches.size()).cuda()
            losses=[]
            m = 0
            #print(x_crop)
            while m < max_iterations:
               
                data = {
                
                'search': mix_patches.detach(),
                'alpha_samll':alpha_samll.detach(),
             
                'alpha_org':alpha_org.detach(),
                #'pert': pert.detach(),
                #'prev_perts': prev_perts.detach(),
                #'label_cls': label_cls.detach(),
                'adv_cls': adv_cls.detach(),
                #'weights':weights.detach(),
                'momen':momen.detach()
                }

                data['alpha_samll'].requires_grad = True
                #data['weights'].requires_grad = True
                alpha_samll, loss = self.attack_once(tracker, data)
                #adv_x_crop = (x_crop.cuda() * alpha_samll ).sum(0)
                #vis.image(adv_x_crop)
                #print(loss,alpha_samll.sum())
                #input()
                losses.append(loss)
                m+=1
            #print(losses)
            
            #alpha_samll = F.softmax(alpha_samll,dim=0)
            adv_x_crop = (mix_patches.cuda() * alpha_samll ).sum(0)
            org_x_crop = (mix_patches.cuda() * alpha_org ).sum(0)
            #print(x_crop)
            #print(loss)
            adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
            #print(adv_x_crop.size(),pert.size(),prev_perts.size())
            #print(adv_x_crop)
            #vis.image((x_crop[0]))
            #vis.image((org_x_crop))
            #vis.image(adv_x_crop)    #permute(2,0,1)
            
            if output_org:
                    advblur_org = tracker.get_orgimg(img1, adv_x_crop.unsqueeze(0), tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
            
            out={
                #'advblur_org':advblur_org,
                'normblur_org':(normblur_org.cpu().detach().numpy())*255,
                'adv_x_crop':adv_x_crop.unsqueeze(0).detach(),
                #'x_crop_normblur':x_crop_normblur.permute(2,0,1).unsqueeze(0)
            }
            return out

        elif mode == 'guo_crop_speed':
            tracker = model_attacked
            X0 = crop[0][0]
            X1 = crop[1][0]
            #print(X0.size())
            
         
            #print(X1.size())
            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))
            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            #print(sam0.size())
            #print(X0.size())
            
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1).cuda()
            #print(cur_offset_input.size())
            
            flow = self.flownets(cur_offset_input)
            temp = flow *flow_scale
            
            flow_full = nn.Upsample(scale_factor=4, mode='bilinear')(temp)
            
            #flow_vis = flip(flow_full,2)   # vis.heatmap 上下方向不对，所以需要翻转过来看
            
            #vis.heatmap(flow_vis[0][0])
            #vis.heatmap(flow_vis[0][1]) 
            #vis.heatmap(flow_new[0][0])
            
            #vis.heatmap(flow_new[0][1])
            #print(flow_full.size())
            ##print(img1.shape)
            
            #vis.image(flow_full[:,0,:,:])
            #vis.image(flow_full[:,1,:,:])
            flow_full_crop = flow_full[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(0,2,3,1)  
            #flow_full[:,:,:,1]=  -flow_full[:,:,:,1]
            sum_img01 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            sum_img10 = torch.zeros(divnum,X0.size(1),X0.size(2),X0.size(3))
            
            theta01 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            theta10 = torch.zeros(divnum,X0.size(2),X0.size(3),2).cuda()
            
            
        
        
           
            #print(flow_full_crop.size())
            
            
            
            
            #vis.image((normblur_org.permute(2,0,1)))
        
            frame_1 =  X1[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255     #.permute(1,2,0)
            frame_0 =  X0[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].unsqueeze(0).cuda()*255 
           
            backbone_feat = tracker.net.extract_backbone(frame_1)                                                                                                                  
            #print(label_in)                                                          
            test_x = tracker.get_classification_features(backbone_feat)
            scores_gt = tracker.classify_target(test_x)  
            

            #print(scores_gt.size()) 
            
            #vis.image(label_in[0])
            #vis.heatmap(scores_gt[0][0])
            adv_cls = self.get_adv_lab(scores_gt).cuda() 

            #vis.heatmap(adv_cls[0][0])
            
            img_sz = frame_0.size(2)
            
            
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
           
            x0input = frame_0.repeat(divnum,1,1,1).detach().cuda()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach().cuda()
      
            alpha_samll = torch.ones(2*divnum,1,img_sz, img_sz).cuda()/(2*divnum)
            alpha_org = torch.ones(2*divnum,1,img_sz, img_sz).cuda()/(2*divnum)
            sx = torch.rand((divnum,1,img_sz,img_sz)).cuda()#.requires_grad = True
            sy = torch.rand((divnum,1,img_sz,img_sz)).cuda()
            #sx = torch.rand((divnum)).cuda()#.requires_grad = True
            #sy = torch.rand((divnum)).cuda()
            sx = F.softmax(sx,dim=0).detach()
            sy = F.softmax(sy,dim=0).detach()
            momen = torch.zeros(frame_1.size()).cuda()
            losses=[]
            m = 0
            firs =True
            last = False
            #print(x_crop)
            while m < max_iterations:
                
                theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                flow_full_crop = flow_full_crop.detach()
                cur_x = frame_1.clone().detach()#/255.0
                idflow = idflow.detach()
                frame_0 = frame_0.detach()
                frame_1 = frame_1.detach()
                sx = sx / sx.sum(0).unsqueeze(0)
                sy = sy / sy.sum(0).unsqueeze(0)
                #print(sx.size(),sx[:,:,3,3])
                if speed :
                    sx.requires_grad = True
                    sy.requires_grad = True
                momen = momen.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
               
                #print(sx,sy)
                
                for i in range(divnum):
                    if speed :
                        theta_fw_x =    (sx[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        theta_fw_y =    (sy[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,1]
                        #theta_fw_x =    (sx[:i+1].sum(0))     * flow_full_crop[:,:,:,0]
                        #theta_fw_y =    (sy[:i+1].sum(0))     * flow_full_crop[:,:,:,1]
                        theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                    else:
                        theta_fw = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                      
                   
                    #input()
                    theta_bw = theta_fw - flow_full_crop 
                    #print(idflow.size(),theta_fw.size())     torch.clamp    ,0,frame_0.size(3)
                    theta01[i]  = (idflow - theta_fw).squeeze(0)
                    theta10[i]  = (idflow - theta_bw).squeeze(0)
                
                theta01[:,:,:,0]=((theta01[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta01[:,:,:,1]=((theta01[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #vis.heatmap(theta01[15,:,:,0])  
                #vis.heatmap(theta01[15,:,:,1])         
                theta10[:,:,:,0]=((theta10[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta10[:,:,:,1]=((theta10[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #theta01.sum().backward() 
                #theta01.sum().backward() 
           
                
                #vis.heatmap(theta01[0,:,:,0])
                warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
           
                #vis.image(warped10[0])
                
                
                mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
                #print(mix_crop.size())
                
                blur_search = (mix_crop * alpha_samll).sum(0).unsqueeze(0)  # 整个search 模糊， 要mask出object
                #vis.heatmap(cur_x[0][1])
                #print(blur_search.size())
                #vis.image(blur_search[0])
                
                #vis.heatmap(blur_search[0][2])
               
                while blur_search.max() >255:
                    blur_search = blur_search * 0.95
                if mask_obj:
                    cur_x[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] = blur_search[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] 
                else:
                    cur_x = blur_search
               
                #print(cur_x.size())
                
      
                backbone_feat = tracker.net.extract_backbone(cur_x)
                #print(backbone_feat)                                                          
                test_x = tracker.net.extract_classification_feat(backbone_feat)
                scores = tracker.net.classifier.classify(tracker.target_filter,test_x)
                    
                #vis.heatmap(scores[0][0])
                #spx_loss = torch.std(sx)
                #spy_loss = torch.std(sy)
                
                
                cls_loss = torch.norm((scores-adv_cls),p=2)
        
                reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
                total_loss = cls_loss + 0.0005 * reg_loss #+ 0 * (spx_loss+ spy_loss)
                #total_loss = test_x.sum()       #backbone_feat['layer2'].sum()  
                
                
                total_loss.backward()
            


                 
                #print(sx[:,:,3,5])
                #print(sx,sy)
                w_grad = -alpha_samll.grad   
                if speed :
                    sx_grad = -sx.grad  
                    sy_grad = -sy.grad
                #
                if MIFGSM:
                    momen = lamb_mome*momen+w_grad/torch.norm(w_grad,1)
                    alpha_samll = (alpha_samll + epsilon *torch.sign(momen)).detach()
                    if speed :
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        #print(sx,sy)
                else:
                    
                    w_grad = torch.sign(w_grad)
                    #w_grad = 0
                    #print(x_grad.sum())
                    alpha_samll = (alpha_samll + epsilon * w_grad).detach()
                    if speed :
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        #sx = F.softmax(sx).detach()
                        #sy = F.softmax(sy).detach()
                        #firs =False
                
                if grad_norm:
                    syg_sum = torch.abs(sy_grad.view(divnum,-1)).sum(1)
                    sxg_sum = torch.abs(sx_grad.view(divnum,-1)).sum(1)
                    sx_min ,xid = torch.topk(sxg_sum,1,largest=False)
                    sy_min ,yid = torch.topk(syg_sum,1,largest=False)
                    #print(xid,yid)

                    res_x = sx.sum(0) - sx[xid]
                    res_y = sy.sum(0) - sy[xid]
                    #print(res_x,res_y)
                    sx[xid] = 1 - res_x
                    sy[xid] = 1 - res_y
                #print(total_loss)
                #print(sx.sum(0))
                
                #print(sx,sy)
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                
                m = m+1
            
            #vis.image(cur_x[0])
            #save_torchimg(cur_x,'dimp')
            #input()
            if output_org:
                    advblur_org = tracker.get_orgimg(img1, adv_x_crop.unsqueeze(0), tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
            #input()
            out={
                #'advblur_org':advblur_org,
                #'normblur_org':(normblur_org.cpu().detach().numpy())*255,
                'adv_x_crop':cur_x
               
                #'x_crop_normblur':x_crop_normblur.permute(2,0,1).unsqueeze(0)
            }
            return out
        
        rth
        return 0
    
 
    def guo_subwin(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[2] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[1] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        #print(im,im.shape , type(im))
        #vis.image(im[0].transpose((2,0,1)))
        print(avg_chans)
    
        n, r, c, k = im.shape
        avg_chans = avg_chans[np.newaxis, np.newaxis,np.newaxis, :]#.repeat(n,0)
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (n,r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.float32)
            te_im[:, top_pad:top_pad + r, left_pad:left_pad + c, : ] = im
            print(avg_chans.shape)
            vis.image(te_im[0].transpose((2,0,1)))
            
            
            if top_pad:
                te_im[:,0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[:,r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:,:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:,:, c + left_pad:, :] = avg_chans
            im_patch = te_im[:,int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[:,int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]
        vis.image(te_im[0].transpose((2,0,1)))
        vis.image(im_patch[0].transpose((2,0,1)))
        f
        if not np.array_equal(model_sz, original_sz):
            limg=[]
            for i in range(n):

                imgi = cv2.resize(im_patch[i], (model_sz, model_sz))
                limg.append(imgi)
            im_patch = numpy.array(limg)
            print(im_patch.shape)
            
        im_patch = im_patch.transpose(0,3,1,2)
        #im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA:
            im_patch = im_patch.cuda()
        return im_patch

    def get_adv_lab(self,scores_gt ):
        sz_score =  scores_gt.size(2) 
        
        score_fa = scores_gt.view(sz_score*sz_score)
        maxid = score_fa.argmax()
        yy = maxid//sz_score
        xx = maxid%sz_score
        
        #print(xx,yy)
        
        adv_no = True 
        while adv_no:
            yp = random.choice((1, -1)) * random.randint(4,sz_score-6)
            xp = random.choice((1, -1)) * random.randint(5,sz_score-6)

            if 1<xx+xp<sz_score-1 and 1<yy+yp<sz_score-1:
                break 
        
        #print(xx+xp,yy+yp)
        
        adv_cls = torch.zeros(scores_gt.size()) 
         
        adv_cls[:,:,yy+yp,xx+xp] = 1
        adv_cls[:,:,yy+yp,xx+xp+1] = 1
        adv_cls[:,:,yy+yp+1,xx+xp+1] = 1
        adv_cls[:,:,yy+yp+1,xx+xp] = 1
        #adv_cls[:,:,yy+yp-1,xx+xp] = 1
        adv_cls[:,:,sz_score//2-1:sz_score//2+1,sz_score//2-1:sz_score//2+1] = -1
        #vis.heatmap(scores_gt[0][0])
        #vis.heatmap(adv_cls[0][0])
        #input()
        '''
        adv_cls[:,:,10,10] = 1
        adv_cls[:,:,10,11] = 1
        adv_cls[:,:,10,9] = 1
        adv_cls[:,:,11,10] = 1
        adv_cls[:,:,9,10] = 1
        #print(adv_cls.size())
        #vis.heatmap(scores_gt[0][0])
        #vis.heatmap(adv_cls[0][0])
        '''
        '''
        while adv_no:
            yp = random.choice((1, -1)) * random.randint(1,9)
            xp = random.choice((1, -1)) * random.randint(1,9)

            if 0<xx+xp<15 and 0<yy+yp<15:
                break 
        adv_cls[:,:,yy+yp,xx+xp] = 1
        while adv_no:
            yp = random.choice((1, -1)) * random.randint(1,9)
            xp = random.choice((1, -1)) * random.randint(1,9)

            if 0<xx+xp<15 and 0<yy+yp<15:
                break 
        adv_cls[:,:,yy+yp,xx+xp] = 1
        '''
        return adv_cls
        
    
    def get_gt_cls(self, tracker, scale_z, outputs):
      
    
        score = tracker._convert_score(outputs['cls'])
        pred_bbox = tracker._convert_bbox(outputs['loc'], tracker.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(tracker.size[0] * scale_z, tracker.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((tracker.size[0] / tracker.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 tracker.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        obj_bbox = pred_bbox[:, best_idx]
        obj_pos = np.array([obj_bbox[0], obj_bbox[1]])

        b, a2, h, w = outputs['cls'].size()
        size = tracker.size
        template_size = np.array([size[0] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1]), \
                                  size[1] + cfg.TRACK.CONTEXT_AMOUNT * (size[0] + size[1])])
        context_size = template_size * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)


        desired_pos = obj_pos
        
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(desired_pos[0], downbound_pos[0], upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1], downbound_pos[1], upbound_pos[1])
        
        desired_bbox = self._get_bbox(desired_pos, size)
        
        
        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])

        self.target_pos = desired_pos

        return adv_cls
    
    def _get_bbox(self, center_pos, shape):
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = center_pos * scale_z
        bbox = center2corner(Center(cx - w / 2, cy - h / 2, w, h))
        return bbox
