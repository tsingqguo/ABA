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
vis=Visdom(env="attacker")

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torchvision
from pysot.models.loss import select_cross_entropy_loss
from pysot.models.backbone.resnet_atrous import ResNet,Bottleneck
from pysot.utils.model_load import load_pretrain
import cv2
from pysot.utils.bbox import center2corner, Center, get_axis_aligned_bbox
from pysot.datasets.anchor_target import AnchorTarget
import warnings
from torch.autograd import Variable



warnings.filterwarnings("ignore")


divnum = 15      
backbone_path = '/workspace/ABA/resnet50.model'
learning_rate = 10000
tv_beta = 2
l1_coeff= 1 
tv_coeff= 0.05
abs_coeff=0.00005
max_iterations = 10
tnum = 10
epsilon =  0.0002    #0.0002  白盒全图
epsilon_s = 0.002
flow_scale = 20.0
xl_lr = 1
yl_lr = 1
vis_img = True
#a_x = 3  # 正为加速
v0_x = 0
v0_y = 0
unit_t = 0.4
lamb_mome = 0.001
x_dev = 0
y_dev = 0
delta_scale = 0.25
output_org = False
mask_obj = False
MIFGSM = True
adv_fix = False
grad_norm = True

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

class attacker(nn.Module):
    def __init__(self):
        super(attacker, self ).__init__() 
        #self.inter_model = DAIN(channel=3,filter_size = 4,timestep=0.5,training=False).eval().cuda()
    
        self.flownets = PWCNet.__dict__['pwc_dc_net']("/workspace/ABA/DAIN/PWCNet/pwc_net.pth.tar").cuda()
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
        zf = data['template_zf'].cuda()
        search = data['search_mix'].cuda()
        alpha_samll = data['alpha_samll'].cuda()
        alpha_org = data['alpha_org'].cuda()
        adv_cls = data['adv_cls'].cuda()
        momen = data['momen'].cuda()
        #prev_perts = data['prev_perts'].cuda()   'alpha_samll':alpha_samll
        #print(search.size(), alpha_samll.size())
        search = (search *alpha_samll).sum(0).unsqueeze(0) 
        track_model = tracker.model
        #pert_sum = prev_perts.sum(0)

        zf_list = []
        if zf.shape[0] > 1:
            for i in range(0, zf.shape[0]):
                zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
        else:
            zf_list = zf
        #print(len(zf_list))
        xf = track_model.backbone(search)
            
        if cfg.ADJUST.ADJUST:
            xf = track_model.neck(xf)
        cls, loc = track_model.rpn_head(zf_list, xf)

        # get loss1
        cls = track_model.log_softmax(cls)
        #print(cls.size())
        #vis.heatmap(cls[0,0,:,:,1])
        #vis.heatmap(cls[0,1,:,:,1])
        #vis.heatmap(cls[0,2,:,:,1])
        
        cls_loss = select_cross_entropy_loss(cls, adv_cls)
        #print(adv_cls.size())

        #c_prev_perts = torch.cat((prev_perts, pert), 0).cuda()
        #t_prev_perts = c_prev_perts.view(c_prev_perts.shape[0]*c_prev_perts.shape[1],
                                         #c_prev_perts.shape[2] * c_prev_perts.shape[3])
        
        reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
        
        total_loss = cls_loss + 0.00005 * reg_loss
        total_loss.backward()
        adv_x = search


        x_grad = -data['alpha_samll'].grad      
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
        return alpha_samll, total_loss, cls
    
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
    
    def get_bigblur(self, full_flow,X0,X1):
        #print(full_flow.size(),X0.size())   #torch.Size([1, 432, 576, 2]) torch.Size([3, 432, 576])
        x0in = X0.repeat(17,1,1,1)
        x1in = X1.repeat(17,1,1,1)
        theta01 = torch.zeros(17,X0.size(2),X0.size(3),2).cuda().detach()
        theta10 = torch.zeros(17,X0.size(2),X0.size(3),2).cuda().detach()
        xl,yl  = torch.meshgrid(torch.Tensor(range(X0.size(2))),torch.Tensor(range(X0.size(3))))
        idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
        for i in range(17):
            theta_fw = (i+1) * full_flow / divnum

            theta_bw = theta_fw - full_flow 
            #print(idflow.size(),theta_fw.size())     torch.clamp    ,0,frame_0.size(3)
            theta01[i]  = (idflow - theta_fw).squeeze(0)
            theta10[i]  = (idflow - theta_bw).squeeze(0)
        theta01[:,:,:,0]=((theta01[:,:,:,0])-X0.size(3)/2)/(X0.size(3)/2)
        theta01[:,:,:,1]=((theta01[:,:,:,1])-X0.size(2)/2)/(X0.size(2)/2)
        #vis.heatmap(theta01[15,:,:,0])  
        #vis.heatmap(theta01[15,:,:,1])         
        theta10[:,:,:,0]=((theta10[:,:,:,0])-X0.size(3)/2)/(X0.size(3)/2)
        theta10[:,:,:,1]=((theta10[:,:,:,1])-X0.size(2)/2)/(X0.size(2)/2)
        #theta01.sum().backward() 
        #theta01.sum().backward() 
           
                
        #vis.heatmap(theta01[0,:,:,0])
        warped01 = F.grid_sample(input=x0in, grid=(theta01), mode='bilinear')
        warped10 = F.grid_sample(input=x1in, grid=(theta10), mode='bilinear')

        mix_crop = torch.cat((warped01,warped10),0).clamp(0,1)*255
                
        org_blur  = (mix_crop).sum(0)/(2*divnum)
        #print(org_blur.size())
        
        #save_torchimg(org_blur.unsqueeze(0),'aaaa')
        
        return org_blur.permute(1,2,0).cpu().detach().numpy() 
    def forward(self,img0,img1,mode ='guo',model_attacked = None,tracker =None, speed = True , blur = None, normblur = False,blur_attack=None,crop = None,flow=None):
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
        elif mode =='obj_speed':
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
                
            full_flow = self.get_flow(X0,X1,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth)   #生成光流
            #print(full_flow.size())     #torch.Size([1, 512, 704, 2])
            # 把光流变形  转numpy 用于切片成  search 大小
            flow_x = full_flow[0,:,:,0].squeeze(0).repeat(3,1,1).permute(1,2,0).detach().cpu().numpy()
            flow_y = full_flow[0,:,:,1].squeeze(0).repeat(3,1,1).permute(1,2,0).detach().cpu().numpy()
            #full_flow = full_flow.permute(0,3,1,2).squeeze(0)
            #print(flow_y.shape) 
            #vis.heatmap(flow_x[:,:,0])
            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            
            flow_y_crop = self.flow_subwindow(flow_y, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            flow_x_crop = self.flow_subwindow(flow_x, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            # 光流的三个通道完全一样
            #print(flow_x_crop.size(),(flow_x_crop[0][0]-flow_x_crop[0][1]).sum())
            flow_full_crop = torch.stack((flow_x_crop[0][0],flow_y_crop[0][0])).permute(1,2,0).unsqueeze(0)
            #print(flow_full_crop.size())
            
            #vis.heatmap(flow_x_crop[0][0])
            
            label_in = img1
            
            label_crop = tracker.get_subwindow(label_in, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            #vis.image(label_crop[0])
            
            outputs = tracker.model.track(label_crop)
            #print(label_crop)
            gt_cls  = self.get_gt_cls(tracker, scale_z, outputs).long().cuda()
            
            adv_cls, same_prev= self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long().cuda()
            
            frame_0 = tracker.get_subwindow(img0, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            frame_1 = label_crop
            
            xl,yl = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            
            if isinstance(tracker.model.zf, list):
                    zf = torch.cat(tracker.model.zf, 0)
            else:
                    zf = tracker.model.zf
            alpha_samll = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            
            alpha_org = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            speed_p = torch.randn(3).cuda()    #  v0_x   v0_y    unit_t 
            speed_p[2] = torch.abs(speed_p[2])
            #print(frame_0)
            
            x0input = frame_0.repeat(divnum,1,1,1).detach()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach()#/255.0
            #flow_mask = torch.zeros(1,255,255)
            obj_w , obj_h = tracker.size
            #print(obj_w , obj_h)
            obj_w*=1.1
            obj_h *=1.1
            losses=[]
            m = 0
            #以下循环 开始BP 攻击参数
            
            
            while m < max_iterations:
                speed_p[2]=0.1
                theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                flow_full_crop = flow_full_crop.detach()
                cur_x = label_crop.clone().detach()#/255.0
                idflow = idflow.detach()
                frame_0 = frame_0.detach()
                frame_1 = frame_1.detach()
                gt_cls = gt_cls.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
                speed_p.requires_grad = True
                v0_x = speed_p[0]
                v0_y = speed_p[1]
                unit_t = speed_p[2]
                a_y = 2*(flow_full_crop[:,:,:,1]-v0_y*divnum*unit_t)/(divnum*unit_t*divnum*unit_t)  # 计算约束的加速度
                a_x = 2*(flow_full_crop[:,:,:,0]-v0_x*divnum*unit_t)/(divnum*unit_t*divnum*unit_t)  # 计算约束的加速度
                for i in range(divnum):
                    theta_fw =    (i+1)     * flow_full_crop  / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                    #theta_fw_y = (i+1) * flow_y_crop[0][0] / divnum
                    theta_bw = (divnum-i-1) * flow_full_crop  / divnum
                    #theta_bw_y = (divnum-i-1) * flow_x_crop[0][0] / divnum
                    #print(theta.size())
                    #print('####################')
                    #print(theta_f[0,213,494,0])
                    #vis.heatmap(theta_f[0,:,:,1])
                    
                    cur_t = (i+1) *unit_t
                    if v0_x != 0:
                        
                        theta_fw[:,:,:,0] = v0_x * cur_t + 0.5* a_x *cur_t*cur_t
                        #theta_f = v0_tune * theta_f +  0.5 * a_tune * theta_f * i/divnum
                        #theta_b = v0_tune * theta_b +  0.5 * a_tune * theta_b * i/divnum
                        
                    if v0_y != 0:
                        
                        #m_v = (torch.abs(theta_f[:,:,:,1]).view(1,-1)).max()
                        #print(m_v/4)
                        theta_fw[:,:,:,1] = v0_y * cur_t + 0.5* a_y *cur_t*cur_t
                        #theta_f[:,:,:,1] = torch.where(flow_full[:,:,:,1]> 0.1, v0_y * cur_t + 0.5* a_y *cur_t*cur_t, theta_f[:,:,:,1]) 
                        #theta_b[:,:,:,1] = flow_full[:,:,:,1] -    theta_f[:,:,:,1]
                    #print(theta.size())
                    #print(idflow.size())
                    
                    #print(theta01.size())
                    #print((idflow - theta).size())
                    #print(a_y.size())
                    
                    #vis.heatmap((v0_y * cur_t + 0.5* a_y *cur_t*cur_t)[0,:,:])
                    #vis.heatmap(theta_f[0,:,:,1])
                    #print(theta_f[0,213,494,0])
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
                #print(speed_p.grad)
                
                #vis.heatmap(theta01[0,:,:,0])
                warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
           
                #vis.image(warped10[0])
                
                
                mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
                
                blur_search = (mix_crop * alpha_samll).sum(0).unsqueeze(0)  # 整个search 模糊， 要mask出object
                #vis.heatmap(cur_x[0][1])
                #vis.heatmap(blur_search[0][1])
                #vis.heatmap(blur_search[0][2])
               
                while blur_search.max() >255:
                    blur_search = blur_search * 0.95
                if mask_obj:
                    cur_x[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] = blur_search[:,:,int(128-obj_h/2):int(128+obj_h/2),int(128-obj_w/2):int(128+obj_w/2)] 
                else:
                    cur_x = blur_search
                #vis.image(blur_search[0])
                #print(blur_search,warped10.size(),blur_search.max(),warped10.min())
                #print(torch.max(cur_x[0]),cur_x[0].size())
                
                #print(cur_x)
                zf_list = []
                if zf.shape[0] > 1:
                    zf = zf.detach()
                    for i in range(0, zf.shape[0]):
                        zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
                else:
                    zf_list = zf
                #print(len(zf_list))
                xf = tracker.model.backbone(cur_x)
                    
                if cfg.ADJUST.ADJUST:
                    xf = tracker.model.neck(xf)
                cls, loc = tracker.model.rpn_head(zf_list, xf)

                # get loss1
                cls = tracker.model.log_softmax(cls)
                cls_loss = select_cross_entropy_loss(cls, adv_cls)    #    gt_cls    adv_cls
                #print(adv_cls.size())

                #c_prev_perts = torch.cat((prev_perts, pert), 0).cuda()
                #t_prev_perts = c_prev_perts.view(c_prev_perts.shape[0]*c_prev_perts.shape[1],
                                                #c_prev_perts.shape[2] * c_prev_perts.shape[3])

                reg_loss = torch.norm(alpha_samll-alpha_org, 2)
                #print(cls.size())
                
                total_loss = cls_loss + 0.3 * reg_loss
                total_loss.backward()
                
                
                #
                #blur_norm_crop = mix_crop.sum(0)/(2*divnum)
                
                alpha_grad = -alpha_samll.grad
                speed_grad = -speed_p.grad
                #print(alpha_grad.size())
                
                #
                alpha_grad = torch.sign(alpha_grad)
                speed_grad = torch.sign(speed_grad)
                
                #print(x_grad.sum())
                alpha_samll = (alpha_samll + epsilon   * alpha_grad).detach()
                speed_p     = (speed_p     + epsilon_s * speed_grad).detach()
                while speed_p[2] <= 0:
                    speed_p[2] += 0.1
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                #vis.image(cur_x[0])
                m = m+1
                #input()
            #print(cur_x)
            '''
            vis.heatmap(cls[0,0,:,:,1])
            vis.heatmap(cls[0,2,:,:,1])
            vis.heatmap(cls[0,1,:,:,1])
            vis.heatmap(cls[0,3,:,:,1])
            vis.heatmap(cls[0,4,:,:,1])
            '''
    
            #vis.image(cur_x[0])
            
            if output_org:
                for i in range(divnum):
                    theta_f = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                    theta_b = (divnum-i-1) * flow_full_crop / divnum
                    theta_b = theta_f - flow_full_crop
                    theta01[i]  = (idflow - theta_f).squeeze(0)
                    theta10[i]  = (idflow - theta_b).squeeze(0)
                
                theta01[:,:,:,0]=((theta01[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta01[:,:,:,1]=((theta01[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #vis.heatmap(theta01[15,:,:,0])  
                #vis.heatmap(theta01[15,:,:,1])         
                theta10[:,:,:,0]=((theta10[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta10[:,:,:,1]=((theta10[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                
                #重采样
                warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
                mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
                
                normblur_crop  = (mix_crop).sum(0)/(2*divnum)#.permute(1,2,0)#.cpu().detach().numpy()
                
                normblur_org = tracker.get_orgimg(img1, normblur_crop, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average).copy()
            else:
                normblur_org =None
            if output_org:
                    advblur_org = tracker.get_orgimg(img1, cur_x, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average).copy()
            #print(normblur_org.shape)
        
            return {
                'normblur_org':normblur_org,
                #'advblur_org':advblur_org,
                'adv_x_crop':cur_x
            }
            
          
        elif mode =='guo':
            tracker = model_attacked
           
            divnum = 15 
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
            
            flow_vis = flip(flow_full,2)   # vis.heatmap 上下方向不对，所以需要翻转过来看
            
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
        
           
            for i in range(divnum):
                theta_f = (i+1) * flow_full / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                theta_b = (divnum-i-1) * flow_full / divnum
                #print(theta.size())
                #print('####################')
               
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
            
                
                
            warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
            out_img = torch.cat((warped01,warped10),0)

            
            
            #print(X1.size())      # [0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
            out_img =  out_img.clamp(0,1.0)[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(0,2,3,1)
            #print(out_img.size())
            normblur_org = out_img.sum(0)/(2*divnum)
            #vis.image((normblur_org.permute(2,0,1)))
            if 1 :
                print('output norm_org')
                #normblur_org = tracker.get_orgimg(img1, x_crop_normblur.permute(2,0,1).unsqueeze(0), tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
                return {'norm_org':normblur_org.detach().cpu().numpy()*255}
            label_in =  X1.clamp(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth].permute(1,2,0)
            #定义 融合参数
        
            alpha = torch.ones(divnum*2,out_img.size(1),out_img.size(2),1).cuda()      #  randn    +1
            #print(out_img.size()) # torch.Size([50, 432, 576, 3])
             
            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            tra_im = out_img.cpu().detach().numpy()*255
            mix_crop = torch.zeros(2*divnum,3,255,255)
            #print(normblur_org.size()) 
            
            '''
            if output_org:
                    normblur_org = tracker.get_orgimg(img1, x_crop_normblur.permute(2,0,1).unsqueeze(0), tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
                    print(normblur_org.shape)
            '''
            for i in range(2*divnum):
                img_mid = tracker.get_subwindow(tra_im[i], tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
                mix_crop[i]=img_mid[0]

            
            #print(tra_im.shape , mix_crop.shape)
            
            #x_crop = self.guo_subwin(out_img.cpu().detach().numpy(), tracker.center_pos,
                                       #cfg.TRACK.INSTANCE_SIZE,
                                       #round(s_x), tracker.channel_average)
            
            x_crop = mix_crop.sum(0).unsqueeze(0)/(2*divnum)
            #print(x_crop)
         
            #print(qq.size())

            #vis.image(qq)
            
            
            
            label_crop = tracker.get_subwindow(label_in.cpu().numpy()*255, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            
            #print(label_in.size())
            #print(label_crop.size())
            #vis.image(label_crop[0])
            

            outputs = tracker.model.track(label_crop)
            #print(outputs['cls'].size())
            
            adv_cls, same_prev= self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()
            
            pert = torch.zeros(x_crop.size()).cuda()
            alpha_samll = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            alpha_org = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            prev_perts = torch.zeros(x_crop.size()).cuda()
            momen = torch.zeros(x_crop.size()).cuda()
            losses=[]
            m = 0
            #print(x_crop)
            while m < max_iterations:
                if isinstance(tracker.model.zf, list):
                    zf = torch.cat(tracker.model.zf, 0)
                else:
                    zf = tracker.model.zf
                
                data = {
                'template_zf': zf.detach(),
                'search': x_crop.detach(),
                'alpha_samll':alpha_samll.detach(),
                'search_mix':mix_crop.detach(),
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
                alpha_samll, loss, update_cls = self.attack_once(tracker, data)
                #adv_x_crop = (x_crop.cuda() * alpha_samll ).sum(0)
                #vis.image(adv_x_crop)
                #print(loss,alpha_samll.sum())
                #input()
                losses.append(loss)
                m+=1
            #print(losses)
            #prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            #alpha_samll = F.softmax(alpha_samll,dim=0)
            adv_x_crop = (x_crop.cuda() * alpha_samll ).sum(0)
            org_x_crop = (x_crop.cuda() * alpha_org ).sum(0)
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
                'advblur_org':advblur_org,
                'normblur_org':(normblur_org.cpu().detach().numpy())*255,
                'adv_x_crop':adv_x_crop.unsqueeze(0).detach(),
                #'x_crop_normblur':x_crop_normblur.permute(2,0,1).unsqueeze(0)
            }
            return out

        elif mode =='guo_speed':
            divnum = 14
            tracker = model_attacked
            #print(img1.shape)
            
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
                
            full_flow = self.get_flow(X0,X1,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth)   #生成光流
            #print(full_flow.size())     #torch.Size([1, 512, 704, 2])
            # 把光流变形  转numpy 用于切片成  search 大小
            flow_x = full_flow[0,:,:,0].squeeze(0).repeat(3,1,1).permute(1,2,0).detach().cpu().numpy()
            flow_y = full_flow[0,:,:,1].squeeze(0).repeat(3,1,1).permute(1,2,0).detach().cpu().numpy()
            #full_flow = full_flow.permute(0,3,1,2).squeeze(0)
            #print(flow_y.shape) 
            #vis.heatmap(flow_x[:,:,0])
            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            
            flow_y_crop = self.flow_subwindow(flow_y, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            flow_x_crop = self.flow_subwindow(flow_x, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            if output_org:
                org_blur_img = self.get_bigblur(full_flow,X0.unsqueeze(0),X1.unsqueeze(0))
            # 光流的三个通道完全一样
            #print(flow_x_crop.size())  # torch.Size([1, 3, 255, 255])

            
            flow_full_crop = torch.stack((flow_x_crop[0][0],flow_y_crop[0][0])).permute(1,2,0).unsqueeze(0)
            #print(flow_full_crop.size())
            
            #vis.heatmap(flow_x_crop[0][0])
            
            label_in = img1
            
            label_crop = tracker.get_subwindow(label_in, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average)
            #vis.image(label_crop[0])
            #fea_label = self.backbone(label_crop)[2]
            #print(fea_label.size())
            
          
            outputs = tracker.model.track(label_crop)
            xf_clean = outputs['xf']
            
            
            #print(label_crop)
            #gt_cls  = self.get_gt_cls(tracker, scale_z, outputs).long().cuda()
            
            adv_cls, same_prev = self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long().cuda()
            
            frame_0 = tracker.get_subwindow(img0, tracker.center_pos, cfg.TRACK.INSTANCE_SIZE, round(s_x), tracker.channel_average)
            frame_1 = label_crop
            
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf

            alpha_samll = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            
            alpha_org   = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            momen = torch.zeros(label_crop.size()).cuda()
            #print(frame_0)
            '''
            sx = torch.rand((divnum,1,255,255)).cuda()#.requires_grad = True
            sy = torch.rand((divnum,1,255,255)).cuda()#.requires_grad = True
            sx = F.softmax(sx,dim=0).detach()
            
            '''
            sx = torch.ones((divnum,1,255,255)).cuda()/divnum
            sy = torch.ones((divnum,1,255,255)).cuda()/divnum
            
            #print(sx.sum(0),sx.sum(0).size())
            #print(sx[:,:,4,3],sx.size())
            
            x0input = frame_0.repeat(divnum,1,1,1).detach()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach()#/255.0
            #flow_mask = torch.zeros(1,255,255)
            obj_w , obj_h = tracker.size
            #print(obj_w , obj_h)
            obj_w*=1.1
            obj_h *=1.1
            losses=[]
            m = 0
            #以下循环 开始BP 攻击参数
            
            #print(sx[:,:,34,64])
            while m < max_iterations:
                print(frame_0.size())
                theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                flow_full_crop = flow_full_crop.detach()
                cur_x = label_crop.clone().detach()#/255.0
                idflow = idflow.detach()
                frame_0 = frame_0.detach()
                frame_1 = frame_1.detach()
                #print('111')
                sx = sx / sx.sum(0).unsqueeze(0)
                sy = sy / sy.sum(0).unsqueeze(0)
                #sx[-1,:,:,:] = 1-sx[:-1,:,:,:].sum(0)
                #sy[-1,:,:,:] = 1-sy[:-1,:,:,:].sum(0)
                #print(sx[:,:,34,64])
                tracker.model.rpn_head.rpn2.cls.act.xt_pre = tracker.model.rpn_head.rpn2.cls.act.xt_pre.detach()
                tracker.model.rpn_head.rpn3.cls.act.xt_pre = tracker.model.rpn_head.rpn3.cls.act.xt_pre.detach()
                tracker.model.rpn_head.rpn4.cls.act.xt_pre = tracker.model.rpn_head.rpn4.cls.act.xt_pre.detach()
                tracker.model.rpn_head.rpn2.cls.act.state = tracker.model.rpn_head.rpn2.cls.act.state.detach()
                tracker.model.rpn_head.rpn3.cls.act.state = tracker.model.rpn_head.rpn3.cls.act.state.detach()
                tracker.model.rpn_head.rpn4.cls.act.state = tracker.model.rpn_head.rpn4.cls.act.state.detach()
                momen = momen.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
                if speed :
                    sx.requires_grad = True
                    sy.requires_grad = True
                #print(sx,sy)
                
                for i in range(divnum):
                    if speed :
                        theta_fw_x =    (sx[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        theta_fw_y =    (sy[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,1]
                        theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                        #print(sx[:i+1,:,4,3].sum(0))
                    else:
                        theta_fw = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        #theta_b = (divnum-i-1) * flow_full / divnum
                    #print(theta_fw.size(), theta_fw[:,:,:,0],theta_fw_x)
                    #print(sx[0:i+1].sum())
                    #theta_fw_y = (i+1) * flow_y_crop[0][0] / divnum
                    #theta_bw = (divnum-i-1) * flow_full_crop  / divnum
                    
                    
                  
                    #print(theta.size())
                    #print(idflow.size())
                    
                    #print(theta01.size())
                    #print((idflow - theta).size())
                    #print(a_y.size())
                    
                    #vis.heatmap((v0_y * cur_t + 0.5* a_y *cur_t*cur_t)[0,:,:])
                    #vis.heatmap(theta_f[0,:,:,1])
                    #print(theta_f[0,213,494,0])
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
                if m==0:
                    norm_blur_crop = mix_crop.sum(0).unsqueeze(0)/(2*divnum)
                
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
                #vis.image(blur_search[0])
                #print(blur_search,warped10.size(),blur_search.max(),warped10.min())
                #print(torch.max(cur_x[0]),cur_x[0].size())
                
                #print(cur_x)
                zf_list = []
                if zf.shape[0] > 1:
                    zf = zf.detach()
                    for i in range(0, zf.shape[0]):
                        zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
                else:
                    zf_list = zf
                #print(len(zf_list))
                xf = tracker.model.backbone(cur_x)
                    
                if cfg.ADJUST.ADJUST:
                    xf = tracker.model.neck(xf)
                cls, loc = tracker.model.rpn_head(zf_list, xf,aux=None)

                # get loss1
                cls = tracker.model.log_softmax(cls)
                #print(cls,cls.size(),adv_cls.size())
                
                cls_loss = select_cross_entropy_loss(cls, adv_cls)    #    gt_cls    adv_cls
                #print(adv_cls.size())


                reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
        
                total_loss = cls_loss + 0.00005 * reg_loss
                total_loss.backward()
               
                

                w_grad = -alpha_samll.grad  
                
                if speed :
                    sx_grad = -sx.grad  
                    sy_grad = -sy.grad
                    #sx_min ,xid = torch.topk(sx.grad,1,dim=0,largest=False)
                    #sy_min ,yid = torch.topk(sx.grad,1,dim=0,largest=False)

                    #print(sx_grad.size(),sx_min.size(),xid)
                    
                #
                if MIFGSM:
                    momen = lamb_mome*momen+w_grad/torch.norm(w_grad,1)
                    
                    alpha_samll = (alpha_samll + epsilon *torch.sign(momen)).detach()
                    if speed :
                        sx = (sx + epsilon_s * torch.sign(sx_grad)).detach()
                        sy = (sy + epsilon_s * torch.sign(sy_grad)).detach()
                        #print(sx,sy)
                else:
                    
                    x_grad = torch.sign(x_grad)
                    #print(x_grad.sum())
                    alpha_samll = alpha_samll + epsilon * x_grad
              
                # 根据导数归一化
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
                    #print(sx.sum(0))
                
                #print(total_loss)
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                #vis.image(cur_x[0])
                m = m+1
                #input()
            #print(xf[0].size())
            
            #print(cur_x.size())
            #save_im = cur_x[0].detach().permute(1,2,0).cpu().numpy()
            #cv2.imwrite('/workspace/ABA/imm.jpg', save_im)
            #vis.image(cur_x[0])
            #vis.heatmap((alpha_samll-alpha_org)[0][0])
            #vis.heatmap((alpha_samll-alpha_org)[2][0])
            #vis.heatmap((alpha_samll-alpha_org)[3][0])
            #save_torchimg(cur_x,'siam')
            if output_org:
                for i in range(divnum):
                    theta_f = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                    theta_b = (divnum-i-1) * flow_full_crop / divnum
                    theta_b = theta_f - flow_full_crop
                    theta01[i]  = (idflow - theta_f).squeeze(0)
                    theta10[i]  = (idflow - theta_b).squeeze(0)
                
                theta01[:,:,:,0]=((theta01[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta01[:,:,:,1]=((theta01[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                #vis.heatmap(theta01[15,:,:,0])  
                #vis.heatmap(theta01[15,:,:,1])         
                theta10[:,:,:,0]=((theta10[:,:,:,0])-frame_0.size(3)/2)/(frame_0.size(3)/2)
                theta10[:,:,:,1]=((theta10[:,:,:,1])-frame_0.size(2)/2)/(frame_0.size(2)/2)
                
                #重采样
                warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
                warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
                mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
                
                normblur_crop  = (mix_crop).sum(0)/(2*divnum)#.permute(1,2,0)#.cpu().detach().numpy()
                
                normblur_org = tracker.get_orgimg(img1, normblur_crop, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average).copy()
            else:
                normblur_crop = None
                normblur_org =None
            if output_org:
                print(org_blur_img.shape)
                advblur_org = tracker.get_orgimg(org_blur_img, cur_x, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average).copy()
                #advblur_org = tracker.get_chengorg(org_blur_img, cur_x)
            #print(normblur_org.shape)
            #vis.image(advblur_org[0])
            #save_im = advblur_org#[0].detach().permute(1,2,0).cpu().numpy()
            #cv2.imwrite('/workspace/ABA/imm.jpg', save_im)
            #print(advblur_org.shape)
            #print(np.mean(advblur_org))
            #save_torchimg(cur_x.int(),'adv') 
            #cv2.imwrite('/workspace/ABA/adv_org.jpg', advblur_org)
            #input()
            return {
                #'normblur_crop':normblur_crop.unsqueeze(0),
                'responsemap_clean':None,
                'responsemap_blur':None,
                'frame':[frame_0,frame_1],
                'warp':[warped01,warped10],
                'flow_full_crop':flow_full_crop ,
                'normblur_org':normblur_org,
                #'advblur_org':advblur_org,
                'adv_x_crop':cur_x,
                'alpha_samll':alpha_samll,
                'xf_adv':xf,
                'xf_clean':xf_clean,
                'outputs_clean':outputs,
                'label_crop':label_crop,
                'norm_blur_crop':norm_blur_crop
                }

        elif mode == 'guo_crop_speed':
            tracker = model_attacked
            X0 = crop[0][0].cuda()
            X1 = crop[1][0].cuda()
            divnum =17
            #print(X0.size(),X1.size())
            
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
                
            flow_full_crop = self.get_flow(X0/255,X1/255,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth)   #生成光流
            #print(flow_full_crop.size())     #torch.Size([1, 512, 704, 2])
            # 把光流变形  转numpy 用于切片成  search 大小
            
            if flow is not None :
                flow_x_crop = flow[0]
                flow_y_crop = flow[1]
                #print(flow_y_crop)
                
                #print(flow_x_crop.size())
                flow_full_crop[0,:,:,0]= flow_x_crop[0][0]
                flow_full_crop[0,:,:,1]= flow_y_crop[0][0]
                
            flow_x_crop = flow_full_crop[0,:,:,0].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
            flow_y_crop = flow_full_crop[0,:,:,1].unsqueeze(0).repeat(3,1,1).unsqueeze(0)
            #print(flow_y_crop.size())
            
            label_in = img1
            
  
            #vis.image(label_crop[0])
            #fea_label = self.backbone(label_crop)[2]
            #print(fea_label.size())
            
            #print(X0.size(),flow_full_crop.size())
            frame_0 = X0.unsqueeze(0)  
            frame_1 = X1.unsqueeze(0)  
            
            outputs = tracker.model.track(frame_1)
            
            #print(label_crop)
            #gt_cls  = self.get_gt_cls(tracker, scale_z, outputs).long().cuda()
            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            adv_cls, same_prev = self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long().cuda()
            
            
            
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf

            alpha_samll = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            
            alpha_org   = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            momen = torch.zeros(frame_1.size()).cuda()
            #print(frame_0)
            '''
            sx = torch.rand((divnum,1,255,255)).cuda()#.requires_grad = True
            sy = torch.rand((divnum,1,255,255)).cuda()#.requires_grad = True
            sx = F.softmax(sx,dim=0).detach()
            
            '''
            sx = torch.ones((divnum,1,255,255)).cuda()/divnum
            sy = torch.ones((divnum,1,255,255)).cuda()/divnum
            
            #print(sx.sum(0),sx.sum(0).size())
            #print(sx[:,:,4,3],sx.size())
            
            x0input = frame_0.repeat(divnum,1,1,1).detach()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach()#/255.0
            #print(x1input)
            
            #flow_mask = torch.zeros(1,255,255)
            obj_w , obj_h = tracker.size
            #print(obj_w , obj_h)
            obj_w*=1.1
            obj_h *=1.1
            losses=[]
            m = 0
            #以下循环 开始BP 攻击参数
            
            #print(sx[:,:,34,64])
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
                #sx[-1,:,:,:] = 1-sx[:-1,:,:,:].sum(0)
                #sy[-1,:,:,:] = 1-sy[:-1,:,:,:].sum(0)
                #print(sx[:,:,34,64])
                
                momen = momen.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
                if speed :
                    sx.requires_grad = True
                    sy.requires_grad = True
                #print(sx,sy)
                
                for i in range(divnum):
                    if speed :
                        theta_fw_x =    (sx[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        theta_fw_y =    (sy[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,1]
                        theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                        #print(theta_fw,flow_full_crop)
                        
                        #print(sx[:i+1,:,4,3].sum(0))
                    else:
                        theta_fw = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        #theta_b = (divnum-i-1) * flow_full / divnum
                    #print(theta_fw.size(), theta_fw[:,:,:,0],theta_fw_x)
                    #print(sx[0:i+1].sum())
                    #theta_fw_y = (i+1) * flow_y_crop[0][0] / divnum
                    #theta_bw = (divnum-i-1) * flow_full_crop  / divnum
                    
                    
                  
                    #print(theta.size())
                    #print(idflow.size())
                    
                    #print(theta01.size())
                    #print((idflow - theta).size())
                    #print(a_y.size())
                    
                    #vis.heatmap((v0_y * cur_t + 0.5* a_y *cur_t*cur_t)[0,:,:])
                    #vis.heatmap(theta_f[0,:,:,1])
                    #print(theta_f[0,213,494,0])
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
                #save_torchimg(warped10,'warp')
                
                
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
                #vis.image(blur_search[0])
                #print(blur_search,warped10.size(),blur_search.max(),warped10.min())
                #print(torch.max(cur_x[0]),cur_x[0].size())
                #save_torchimg(cur_x,'siam')
                
                #print(cur_x)
                zf_list = []
                if zf.shape[0] > 1:
                    zf = zf.detach()
                    for i in range(0, zf.shape[0]):
                        zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
                else:
                    zf_list = zf
                #print(len(zf_list))
                xf = tracker.model.backbone(cur_x)
                    
                if cfg.ADJUST.ADJUST:
                    xf = tracker.model.neck(xf)
                cls, loc = tracker.model.rpn_head(zf_list, xf)

                # get loss1
                cls = tracker.model.log_softmax(cls)
                cls_loss = select_cross_entropy_loss(cls, adv_cls)    #    gt_cls    adv_cls
                #print(adv_cls.size())


                reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
        
                total_loss = cls_loss + 0.00005 * reg_loss
                total_loss.backward()
               
                

                w_grad = -alpha_samll.grad  
                
                if speed :
                    sx_grad = -sx.grad  
                    sy_grad = -sy.grad
                    #sx_min ,xid = torch.topk(sx.grad,1,dim=0,largest=False)
                    #sy_min ,yid = torch.topk(sx.grad,1,dim=0,largest=False)

                    #print(sx_grad.size(),sx_min.size(),xid)
                    
                #
                if MIFGSM:
                    momen = lamb_mome*momen+w_grad/torch.norm(w_grad,1)
                    
                    alpha_samll = (alpha_samll + epsilon *torch.sign(momen)).detach()
                    if speed :
                        sx = (sx + 0.0001 * torch.sign(sx_grad)).detach()
                        sy = (sy + 0.0001 * torch.sign(sy_grad)).detach()
                        #print(sx,sy)
                else:
                    
                    x_grad = torch.sign(x_grad)
                    #print(x_grad.sum())
                    alpha_samll = alpha_samll + epsilon * x_grad
              
                # 根据导数归一化
                if grad_norm and speed :
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
                    #print(sx.sum(0))
                
                #print(total_loss)
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                #vis.image(cur_x[0])
                m = m+1
                #input()
            #print(cur_x.size())
            #save_im = cur_x[0].detach().permute(1,2,0).cpu().numpy()
            #cv2.imwrite('/workspace/ABA/imm.jpg', save_im)
            #vis.image(cur_x[0])
            #vis.heatmap((alpha_samll-alpha_org)[0][0])
            #vis.heatmap((alpha_samll-alpha_org)[2][0])
            #vis.heatmap((alpha_samll-alpha_org)[3][0])
            #save_torchimg(cur_x,'siam')
            
            #print(normblur_org.shape)
            #vis.image(advblur_org[0])
            #save_im = advblur_org#[0].detach().permute(1,2,0).cpu().numpy()
            #cv2.imwrite('/workspace/ABA/imm.jpg', save_im)
            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            advblur_org = tracker.get_orgimg(img1, cur_x, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average).copy()
            #print(advblur_org.shape,type(advblur_org))
            
            #save_torchimg(advblur_org,'tmp')
            #vis.image(advblur_org.transpose(2,0,1))
            
            return {
                    
           
                'advblur_org':advblur_org,
                'flow_crop':[flow_x_crop,flow_y_crop],
                'adv_x_crop':cur_x,
                'alpha_samll':alpha_samll
                
            }

        elif mode =='onestep':  
            divnum =17
            #max_iterations = 10 
            data = img0
            template = data['template'].cuda()
            X1 = data['search'].squeeze(0).cuda()
            X0 =  data['search2'].squeeze(0).cuda()
            label_cls = data['label_cls'].cuda()
            tracker = model_attacked
            #vis.image(X0)
            #vis.image(X1)
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
            #print(intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom)   
            #print(X0.size(),X1.size())
            
            full_flow = self.get_flow(X0/255,X1/255,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth)   #生成光流
            #print(full_flow.size())     #torch.Size([1, 512, 704, 2])
            
            # 把光流变形  转numpy 用于切片成  search 大小
         
            #full_flow = full_flow.permute(0,3,1,2).squeeze(0)
            #print(flow_y.shape) 
            #vis.heatmap(flow_x[:,:,0])
            

            # 光流的三个通道完全一样
            #print(flow_x_crop.size())  # torch.Size([1, 3, 255, 255])

            #vis.heatmap(full_flow[0,:,:,0])
            flow_full_crop = full_flow
            #print(flow_full_crop.size())
            zf = tracker.backbone(template)
            xf = tracker.backbone(X1.unsqueeze(0))
            if cfg.MASK.MASK:
                zf = zf[-1]
                tracker.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                zf = tracker.neck(zf)
                xf = tracker.neck(xf)
            #self.zf = zf
            cls, loc = tracker.rpn_head(zf, xf)
            #vis.heatmap(cls[0][0])
            #计算clean图的heatmap高亮点
            #cls = tracker.log_softmax(cls)
            score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
            cls= F.softmax(score, dim=1).data[:, 1].view(-1)
            #print(cls.size())
            #vis.heatmap(cls.view(5,25,25)[0])
            
            v,id = torch.max(cls,0)
            #print(v,id)
            mapi = id//(25*25)
            pp = id%(25*25)
            yy = pp//25
            xx = pp%25
            #print(pp,yy,xx)
            #vis.heatmap(cls.view(5,25,25)[mapi])
            #print(torch.rand((1)),torch.rand((1)),torch.rand((1)),torch.rand((1)),torch.rand((1)))
            
            while 1 :
                xadv = int(torch.rand((1)) * 23)
                yadv = int(torch.rand((1)) * 23)
                if ((xadv-xx)**2 + (yadv-yy)**2) > 6**2:
                    break
            #print(yadv ,xadv)
            adv_cls = torch.zeros((5,25,25))
            adv_cls[:,yadv:yadv+2,xadv:xadv+2] = 1
            #vis.heatmap(adv_cls[0])
        
            '''
            vis.heatmap(cls[0])
            vis.heatmap(cls[1])
            vis.heatmap(cls[2])
            vis.heatmap(cls[3])
            vis.heatmap(cls[4])
            '''
            #vis.heatmap(flow_x_crop[0][0])
            
            
       
            
      
            
            adv_cls = adv_cls.long().cuda()
            
            frame_0 = X0.unsqueeze(0)
            frame_1 = X1.unsqueeze(0)
            
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            
            

            alpha_samll = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            
            alpha_org   = torch.ones(2*divnum,1,255,255).cuda()/(2*divnum)
            
            momen = torch.zeros(alpha_samll.size()).cuda()
            #print(frame_0)
            '''
            sx = torch.rand((divnum,1,255,255)).cuda()#.requires_grad = True
            sy = torch.rand((divnum,1,255,255)).cuda()#.requires_grad = True
            sx = F.softmax(sx,dim=0).detach()
            sy = F.softmax(sy,dim=0).detach()
            '''
            sx = torch.ones((divnum,1,255,255)).cuda()/divnum
            sy = torch.ones((divnum,1,255,255)).cuda()/divnum
            
            #print(sx.sum(0),sx.sum(0).size())
            #print(sx[:,:,4,3],sx.size())
            
            x0input = frame_0.repeat(divnum,1,1,1).detach()#/255.0
            x1input = frame_1.repeat(divnum,1,1,1).detach()#/255.0
            #flow_mask = torch.zeros(1,255,255)
           
        
            
            losses=[]
            m = 0
            #以下循环 开始BP 攻击参数
            
            #print(sx[:,:,34,64])
            while m < max_iterations:
                
                theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda().detach()
                flow_full_crop = flow_full_crop.detach()
                #cur_x = label_crop.clone().detach()#/255.0
                idflow = idflow.detach()
                frame_0 = frame_0.detach()
                frame_1 = frame_1.detach()
                #sx = sx / sx.sum(0).unsqueeze(0)
                #sy = sy / sy.sum(0).unsqueeze(0)
                sx[-1,:,:,:] = 1-sx[:-1,:,:,:].sum(0)
                sy[-1,:,:,:] = 1-sy[:-1,:,:,:].sum(0)
                #print(sx[:,:,34,64])
                
                momen = momen.detach()
                adv_cls = adv_cls.detach()
                alpha_samll.requires_grad = True
                #sx.requires_grad = True
                #sy.requires_grad = True
                #print(sx,sy)
                
                for i in range(divnum):
                    if speed :
                        gd
                        theta_fw_x =    (sx[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        theta_fw_y =    (sy[:i+1,:,:,:].sum(0))     * flow_full_crop[:,:,:,1]
                        theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                        #print(sx[:i+1,:,4,3].sum(0))
                    else:
                        theta_fw = (i+1) * flow_full_crop / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                        #theta_b = (divnum-i-1) * flow_full / divnum
                    #print(theta_fw.size(), theta_fw[:,:,:,0],theta_fw_x)
                    #print(sx[0:i+1].sum())
                    #theta_fw_y = (i+1) * flow_y_crop[0][0] / divnum
                    #theta_bw = (divnum-i-1) * flow_full_crop  / divnum
                    
                    
                  
                    #print(theta.size())
                    #print(idflow.size())
                    
                    #print(theta01.size())
                    #print((idflow - theta).size())
                    #print(a_y.size())
                    
                    #vis.heatmap((v0_y * cur_t + 0.5* a_y *cur_t*cur_t)[0,:,:])
                    #vis.heatmap(theta_f[0,:,:,1])
                    #print(theta_f[0,213,494,0])
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
                #vis.image(blur_search[0])
                #print(blur_search,warped10.size(),blur_search.max(),warped10.min())
                #print(torch.max(cur_x[0]),cur_x[0].size())
                
                #print(cur_x)
                
                #print(len(zf_list))
                #xf = tracker.model.backbone(cur_x)
                xf = tracker.backbone(cur_x)
                
                if cfg.ADJUST.ADJUST:
                    #zf = tracker.neck(zf)
                    xf = tracker.neck(xf)
                #self.zf = zf
                cls, loc = tracker.rpn_head(zf, xf)

                #计算clean图的heatmap高亮点
                #cls = tracker.log_softmax(cls)
                #score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
                #cls= F.softmax(score, dim=1).data[:, 1].view(-1)   

                
               

                # get loss1
                cls = tracker.log_softmax(cls)
                cls_loss = select_cross_entropy_loss(cls, adv_cls)    #    gt_cls    adv_cls
                #print(adv_cls.size())


                reg_loss = torch.norm(alpha_samll-alpha_org, 2,1).sum()
        
                total_loss = cls_loss + 0.0005 * reg_loss
                total_loss.backward()
               
                

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
                    
                    x_grad = torch.sign(x_grad)
                    #print(x_grad.sum())
                    alpha_samll = alpha_samll + epsilon * x_grad
              

                #print(total_loss)
                #print(speed_p)
                #print(cls_loss,0.3 * reg_loss)
                #vis.image(cur_x[0])
                m = m+1
                #input()
            #print(cur_x.size())
            #save_im = cur_x[0].detach().permute(1,2,0).cpu().numpy()
            #cv2.imwrite('/workspace/ABA/imm.jpg', save_im)
            #vis.image(cur_x[0])
            
            
            
            return {
                'adv_x_crop':cur_x.detach(),
                'zf': zf ,#.detach()
                'adv_cls':adv_cls
            }
        elif mode =='onestep_cheng':
            speed = False
            data = img0
            template = data['template'].cuda()
            X1 = data['search'].squeeze(0).cuda()
            X0 =  data['search2'].squeeze(0).cuda()
            label_cls = data['label_cls'].cuda()
            tracker = model_attacked
            #vis.image(X0)
            #vis.image(X1)
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
            #print(intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom)   
            #print(X0.size(),X1.size())
            
            full_flow = self.get_flow(X0/255,X1/255,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth)   #生成光流
            divnum = blur.divnum
            #print(full_flow.size())   #torch.Size([1, 255, 255, 2])
            

            flow_full = full_flow
            #flow_full[:,:,:,1]=  -flow_full[:,:,:,1]
            
            zf = tracker.backbone(template)
            xf = tracker.backbone(X1.unsqueeze(0))
            if cfg.MASK.MASK:
                zf = zf[-1]
                tracker.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                zf = tracker.neck(zf)
                xf = tracker.neck(xf)
            #self.zf = zf
            cls, loc = tracker.rpn_head(zf, xf)

            #计算clean图的heatmap高亮点
            #cls = tracker.log_softmax(cls)
            score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
            cls= F.softmax(score, dim=1).data[:, 1].view(-1)
            #print(cls.size())
            v,id = torch.max(cls,0)
            #print(v,id)
            mapi = id//(25*25)
            pp = id%(25*25)
            yy = pp//25
            xx = pp%25
            #print(pp,yy,xx)
            #vis.heatmap(cls.view(5,25,25)[mapi])
            #print(torch.rand((1)),torch.rand((1)),torch.rand((1)),torch.rand((1)),torch.rand((1)))
            
            while 1 :
                xadv = int(torch.rand((1)) * 23)
                yadv = int(torch.rand((1)) * 23)
                if ((xadv-xx)**2 + (yadv-yy)**2) > 10**2:
                    break
            #print(yadv ,xadv)
            adv_cls = torch.zeros((5,25,25))
            adv_cls[:,yadv:yadv+2,xadv:xadv+2] = 1
            #vis.heatmap(adv_cls[0])
            
            '''
            vis.heatmap(cls[0])
            vis.heatmap(cls[1])
            vis.heatmap(cls[2])
            vis.heatmap(cls[3])
            vis.heatmap(cls[4])
            '''
            #vis.heatmap(flow_x_crop[0][0])
            
            
       
            
      
            
            adv_cls = adv_cls.long().cuda()
            
            frame_0 = X0.unsqueeze(0)
            frame_1 = X1.unsqueeze(0)
            X0 = frame_0
            X1 = frame_1
            xl,yl  = torch.meshgrid(torch.Tensor(range(frame_0.size(2))),torch.Tensor(range(frame_0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            theta01 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda()
            theta10 = torch.zeros(divnum,frame_0.size(2),frame_0.size(3),2).cuda()
            
            #print(idflow.size())
            #idflow[0,:,:,1] = flip(idflow[0,:,:,1],0)   #光流绝对坐标翻转， 左上角为0,0
            #vis.heatmap(idflow[0,:,:,0])
            #vis.heatmap(idflow[0,:,:,1])
            #print(idflow[0,:,:,1])
        
           
            for i in range(divnum):
                theta_f = (i+1) * flow_full / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                theta_b = (divnum-i-1) * flow_full / divnum

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
            
            
            warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
            out_img = torch.cat((warped01,warped10),0)

            #print(X1.size())      # [0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]
            out_img =  out_img.clamp(0,255.0)[:, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth]#.permute(0,2,3,1)
            
            normblur_org = (out_img.sum(0)/(2*divnum)).detach()
            #print(normblur_org.size(),out_img.size())
            #vis.image(normblur_org)

            #print(X0.size(),X1.size())
            img_mix = torch.stack((X0,X1)).view(1,6,255,255).detach()
            blur_one ,  w_out1 ,  w_out2 = blur(img_mix,flow_full)

            #print(blur_one.size() ,  w_out1.size() ,  w_out2.size())

            xf = tracker.backbone(blur_one)
                
            if cfg.ADJUST.ADJUST:
                    #zf = tracker.neck(zf)
                    xf = tracker.neck(xf)
              
            cls, loc = tracker.rpn_head(zf, xf)

                
                
               

             
            cls = tracker.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, adv_cls)

            real_loss = torch.norm((blur_one-normblur_org),2)
            print(cls_loss,real_loss)
            loss = cls_loss #+ 0.0001   *real_loss
            
            return {
                'loss':loss
                } 



    def train_onestep(self,img0, img1, tracker =None, blur_attack=None, attacker_out=None):
            blur_attack = blur_onestep(6,17).train().cuda()
            w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            x0 = tracker.get_subwindow(img0, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            x1 = tracker.get_subwindow(img1, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf
            outputs = tracker.model.track(x1)
            #print(label_crop)
            #gt_cls  = self.get_gt_cls(tracker, scale_z, outputs).long().cuda()
            
            adv_cls, same_prev = self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long().cuda()
            #vis.heatmap(adv_cls[0][0])
            #adv_cls*=0
            #out_ack = ack(img0=X0,img1=X1,mode = 'guo_speed' ,model_attacked = tracker,speed = False)               
            img_mix = torch.stack((x0,x1)).view(1,6,255,255)
            optimizer = torch.optim.SGD(blur_attack.parameters(), lr = 0.2)
            for idd in range(20):
                blur_one ,  w_out1 ,  w_out2  = blur_attack(img_mix)
                w_norm_lab = torch.ones_like(w_out1)/17
                loss_reg1 = torch.norm(w_out1-w_norm_lab, 2,1).sum()
                loss_reg2 = torch.norm(w_out2-w_norm_lab, 2,1).sum()
                 
                #vis.heatmap(w_out1[0][0])
                #print(w_out1,w_out2)
                '''
                xf = tracker.model.backbone(blur_one)
                zf_list = []
                if zf.shape[0] > 1:
                    zf = zf.detach()
                    for i in range(0, zf.shape[0]):
                        zf_list.append(zf[i, :, :, :].resize_(1, zf.shape[1], zf.shape[2], zf.shape[3]))
                else:
                    zf_list = zf

                if cfg.ADJUST.ADJUST:
                    #zf = tracker.neck(zf)
                    xf = tracker.model.neck(xf)
                
                cls, loc = tracker.model.rpn_head(zf_list, xf)

                
                    
                

                
                cls = tracker.model.log_softmax(cls)
                cls_loss = select_cross_entropy_loss(cls, adv_cls) 
                #print(cls.size())
                

                loss = cls_loss   + 0.0000 * (loss_reg1+loss_reg2)

                print(cls_loss,(loss_reg1+loss_reg2))
                '''
                w_mix = torch.cat((w_out1,w_out2),dim=0)
                #print(w_mix.size())
                
                loss = torch.norm(attacker_out['alpha_samll']-w_mix)
                print(loss)
                if is_valid_number(loss.data.item()):
                    optimizer.zero_grad()
                    loss.backward()
                    '''
                    for p in blur_attack.parameters():
                        if p.grad is not None:
                            p.grad = torch.sign(p.grad)
                    '''
                    optimizer.step()
                #input()
            #vis.image(blur_one[0])
            print(w_out1[:,0,4,64])
            #vis.heatmap(cls[0,0,:,:,1])
            #vis.heatmap(cls[0,0,:,:,0])
            vis.heatmap(w_out1[0][0])
            return blur_one
        
        
    
  

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

    def ua_label(self, tracker, scale_z, outputs):

        score = tracker._convert_score(outputs['cls'])
        #print(outputs['loc'].size(),tracker.anchors.shape)
        
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

        same_prev = False
        # validate delta meets the requirement of obj
        
        if self.prev_delta is not None:
            diff_pos = np.abs(self.prev_delta - obj_pos)
            if (size[0] // 2 < diff_pos[0] and diff_pos[0] < context_size[0] // 2) \
                    and (size[1] // 2 < diff_pos[1] and diff_pos[1] < context_size[1] // 2):
                delta = self.prev_delta
                same_prev = True
            else:
                delta = []
                delta.append(random.choice((1, -1)) * random.randint(size[0] // 2, context_size[0] // 2))
                delta.append(random.choice((1, -1)) * random.randint(size[1] // 2, context_size[1] // 2))
                delta = obj_pos + np.array(delta)* delta_scale
                self.prev_delta = delta
        else:
        
        
        
            delta = []
            delta.append(random.choice((1, -1)) * random.randint(size[0] // 2, context_size[0] // 2 - size[0] // 2))
            delta.append(random.choice((1, -1)) * random.randint(size[1] // 2, context_size[1] // 2 - size[1] // 2))
            delta = obj_pos + np.array(delta)* delta_scale
            self.prev_delta = delta
        
        
        #print(obj_pos,np.array(delta),delta_scale)
        
        desired_pos = context_size / 2 + delta 
        #print(desired_pos, delta)
        #input()
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(desired_pos[0], downbound_pos[0], upbound_pos[0])
        desired_pos[1] = np.clip(desired_pos[1], downbound_pos[1], upbound_pos[1])
        
        desired_bbox = self._get_bbox(desired_pos, size)
        
        
        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        #print(len(results), results[2].shape, results[3].shape)
        
        overlap = results[3]
        #print(overlap)
    
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        #print(adv_cls.shape, max_pos)
        '''
        
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        #print((max_pos[1]+1 < 25)[0])
        #print(max_pos[1],max_pos[2])
        m1 = max_pos[1][0]
        m2 = max_pos[2][0]
        #print(m1,m2)
        
        if m1+1 < 25 and (m2+1<25):
            
            adv_cls[:, m1+1, m2+1] = 1
        if (m1-1>0) and (m2-1>0):
            adv_cls[:, m1-1, m2-1] = 1
        '''
        adv_no = True 
        xx=12
        yy=12
        while adv_no:
            yp = random.choice((1, -1)) * random.randint(5,8)
            xp = random.choice((1, -1)) *random.randint(6,11)

            if 1<xx+xp<23 and 1<yy+yp<23:
                break 
        adv_pos = [14,22]
        if adv_fix :
            xp = 10   
            yp = 2
        adv_cls[:, yy+yp,xx+xp] = 1
        adv_cls[:, yy+yp+1,xx+xp] = 1
        adv_cls[:, yy+yp,xx+xp+1] = 1
        adv_cls[:, yy+yp+1,xx+xp+1] = 1
        adv_cls[:, 11:13, 11:13] = -1
        
        adv_cls = adv_cls.view(1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2])
        #print(adv_cls.size())
        #vis.heatmap(adv_cls[0][1])
        #input()
        self.target_pos = desired_pos

        return adv_cls,same_prev
    
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

    #def get_orgimg(self, x_crop,img):
        

def save_torchimg(img,name):
    if not isinstance(img,numpy.ndarray):
        img = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/workspace/ABA/{}.jpg'.format(name), img)



def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)









