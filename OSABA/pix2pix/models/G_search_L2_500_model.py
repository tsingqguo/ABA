import torch
from .base_model import BaseModel
from . import networks

'''siamrpn++'''
from OSABA.siamRPNPP import SiamRPNPP
from OSABA.data_utils import normalize
import torch.nn as nn
from DAIN import PWCNet
import warnings
import random
import cv2
from pysot.models.loss import select_cross_entropy_loss
from torch.nn import functional as F
'''hyper-parameters, which may need to be tuned'''
cls_thres = 0.7
warnings.filterwarnings("ignore")
from  visdom import Visdom
vis=Visdom(env="csa")

myadvloss = True
flow_test = True
W_test = True
def save_torchimg(img,name):
    save_im = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/cheng/CSA-master/{}.jpg'.format(name), save_im)


class GsearchL2500Model(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L2', type=float, default=500, help='weight for L2 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'fool' , 'W_L2' , 'fool_cheng' ,'W0_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['search_clean_vis','search_adv_vis']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
    
        opt.netG = 'cheng_unet_256'
        self.netG = networks.define_G(6*17, 2*17, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight = 0.1
            self.margin = -5
            self.weight = self.init_weight
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        '''siamrpn++'''
        
        self.siam = SiamRPNPP(mod='')   # used in training m2 run mobilenet   None resnet50
       
        self.blur = Blur()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.template = input[0].squeeze(0).cuda()  # pytorch tensor, shape=(1,3,127,127)
        self.search_clean255_0 = input[1].squeeze(0).cuda() # pytorch tensor, shape=(N,3,255,255) [0,255]
        self.search_clean0 = normalize(self.search_clean255_0)
        self.search_clean255_1 = input[2].squeeze(0).cuda() # pytorch tensor, shape=(N,3,255,255) [0,255]
        self.search_clean1 = normalize(self.search_clean255_1)
        self.num_search = self.search_clean1.size(0)
        # print('clean image shape:',self.init_frame_clean.size())


    def forward(self,target_sz=(255,255),img2blur= None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''resize to (512,512)'''
        if img2blur is not None :
            #print(img2blur[1])
            
            self.search_clean255_0 = img2blur[1] 
            self.search_clean255_1 = img2blur[0] 
            self.search_blur_n , self.mix_img = self.blur.get_blur_norm(self.search_clean255_0,self.search_clean255_1)
        mix_img = torch.nn.functional.interpolate(self.mix_img, size=(256,256), mode='bilinear')
        #print(mix_img)
        mix_img = normalize(mix_img)
        #print(mix_img.size())
        
        #search512_clean0 = torch.nn.functional.interpolate(self.search_clean0, size=(256,256), mode='bilinear')
        #search512_clean1 = torch.nn.functional.interpolate(self.search_clean1, size=(256,256), mode='bilinear')
        #search512_clean_mix = torch.cat((search512_clean0,search512_clean1),1)
        #print(search512_clean_mix.size())
        flow_x ,flow_y, self.W_0 = self.netG(mix_img)
        #print( flow_x.size(), self.W_0.size())
        #print(flow_2.is_cuda)
        
        #for t in range(2):
            #print()
          
            #vis.heatmap(self.W_0[0][t])
            #vis.heatmap(self.W_0[0][t])
            #input()
    
        self.W = 1 + self.W_0  # Residual form: G(A)+A
        #W = F.softmax(W,dim=1)
        #print(self.netG(search512_clean1))
        self.W      = torch.nn.functional.interpolate(self.W, size=target_sz, mode='bilinear')
        #print(W_0.size())
        flow_x = torch.nn.functional.interpolate(flow_x, size=target_sz, mode='bilinear')
        flow_y = torch.nn.functional.interpolate(flow_y, size=target_sz, mode='bilinear')
        #print(target_sz)
        
        #flow_1 = torch.nn.functional.interpolate(flow_1, size=target_sz, mode='bilinear')
        #flow_2 = torch.nn.functional.interpolate(flow_2, size=target_sz, mode='bilinear')
        self.blur_adv , self.loss_flow = self.blur(self.search_clean255_0,self.search_clean255_1,self.W,flow_x,flow_y)
        #print(self.blur_adv)
        #vis.image(self.blur_adv[0])
        #print(self.blur_adv.is_cuda)
        
        self.search_adv255 = self.blur_adv.cuda()
        self.blur_adv_11 = normalize(self.blur_adv.cuda())
        '''Then crop back to (255,255)'''
        #self.search_adv1 = torch.nn.functional.interpolate(search512_adv1, size=target_sz, mode='bilinear')
        # self.diff = self.search_adv1 - self.search_clean1
        # print(torch.mean(torch.abs(self.diff)))
        #self.search_adv255 = self.search_adv1 * 127.5 + 127.5
        '''for visualization'''
        self.search_clean_vis = self.search_clean1[0:1]
        self.search_adv_vis = self.blur_adv[0:1]
    def transform(self,patch_clean1,target_sz=(255,255)):
        '''resize to (512,512)'''
        patch512_clean1 = torch.nn.functional.interpolate(patch_clean1, size=(512, 512), mode='bilinear')
        patch512_adv1 = patch512_clean1 + self.netG(patch512_clean1)  # Residual form: G(A)+A

        patch_adv1 = torch.nn.functional.interpolate(patch512_adv1, size=target_sz, mode='bilinear')
        patch_adv255 = patch_adv1 * 127.5 + 127.5
        return patch_adv255
    def backward_G(self):
        """Calculate GAN and L2 loss for the generator"""
        # Second, G(A) = B

        self.loss_G_L2 = self.criterionL2(self.blur_adv_11, self.search_blur_n_11) * self.opt.lambda_L2
        num_attention = int(torch.sum(self.score_maps_clean > cls_thres))
        if num_attention > 0:
            if myadvloss:
                w_norm = torch.ones_like(self.W)
                #print(self.score_maps_adv.size())
                #adv_lab =  get_advlabel()
                #self.myscore = self.siam.model.log_softmax(self.myscore)
                #print(self.score_maps_adv.size(),adv_lab.size())
                self.loss_W_L2 = torch.norm(self.W-w_norm, 2) * 0.005
                kx = torch.Tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).repeat(1,34,1,1).cuda()
                ky = torch.Tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).repeat(1,34,1,1).cuda()
                gx = F.conv2d(self.W_0,kx)
                gy = F.conv2d(self.W_0,ky)
                #print(gx)
                
                
                self.loss_W0_L2 = torch.norm(gx+gy, 2) * 0.004
                #print(self.loss_W0_L2)
                
                #adv_lab =  get_advlabel(self.score_maps_clean)
                #self.loss_fool = 0
                #for i in range(adv_lab.size(0)):
                   #self.loss_fool+= select_cross_entropy_loss(self.score_maps_adv[i].unsqueeze(0),adv_lab[i].unsqueeze(0))
                #self.loss_cross_entropy = select_cross_entropy_loss(self.myscore,adv_lab) *  1
                #print(self.loss_fool)
                
            #print(self.score_maps_clean.size(),self.score_maps_adv.size())
            
            score_map_adv_att = self.score_maps_adv[self.score_maps_clean > cls_thres]
            #print(score_map_adv_att.size(),self.opt.lambda_L2)
            adv_id  = torch.rand(20) * self.score_maps_adv.size(0)
            score_map_adv_rand = self.score_maps_adv[adv_id.long()]
            #print(score_map_adv_rand.size())
            
            self.loss_fool_cheng = torch.mean(torch.clamp(score_map_adv_rand[:, 1] - score_map_adv_rand[:, 0], min=self.margin)) * self.weight
            self.loss_fool = torch.mean(torch.clamp(score_map_adv_att[:, 1] - score_map_adv_att[:, 0], min=self.margin)) * self.weight
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L2 + self.loss_fool  +  self.loss_W_L2 + self.loss_fool_cheng + self.loss_W0_L2
        else:
            self.loss_G = self.loss_G_L2
        #print(self.loss_G)
        
        self.loss_G.backward()

    def optimize_parameters(self):
        '''One forward & backward pass. One update of parameters'''
        '''forward pass'''
        self.search_blur_n , self.mix_img = self.blur.get_blur_norm(self.search_clean255_0,self.search_clean255_1)
        #print(self.search_blur_n.is_cuda)
        #print(self.search_blur_n)
        self.search_blur_n_11 = normalize(self.search_blur_n)  #归一化到 -1 1
        # 1. predict with clean template
        with torch.no_grad():
            self.siam.model.template(self.template)
            #print(self.search_clean255.size(),self.template.size())
            
            self.score_maps_clean , _  = self.siam.get_heat_map(self.search_blur_n,softmax=True,dir=True)#(5HWN,),with softmax
        # 2. adversarial attack with GAN
        self.forward()  # compute fake image
        # 3. predict with adversarial template
        self.score_maps_adv , self.myscore = self.siam.get_heat_map(self.search_adv255,softmax=False,dir=True)#(5HWN,2),without softmax
        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        

class Blur(nn.Module):
    def __init__(self):
        super(Blur, self ).__init__() 
        #self.inter_model = DAIN(channel=3,filter_size = 4,timestep=0.5,training=False).eval().cuda()
    
        self.flownets = PWCNet.__dict__['pwc_dc_net']("/workspace/ABA/DAIN/PWCNet/pwc_net.pth.tar").cuda()
        self.flow_dif = torch.nn.MSELoss(reduce = True,size_average = True)
    def get_flow(self,X0,X1,intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth):
            
                
                    
            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                
            
            #print(sam0.size())
            #print(X0.size())
            X0 = X0.unsqueeze(0)
            X1 = X1.unsqueeze(0)
            X0 = pader(X0)
            X1 = pader(X1)

            cur_offset_input = torch.cat((X0, X1), dim=1)/255
            #print(cur_offset_input.size())
            
            
            flow = self.flownets(cur_offset_input)
            temp = flow *20
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
            #print(flow_full.size())
            
            return  flow_full

    def get_blur_norm(self,X0,X1):
        divnum = 17
        self.divnum = divnum
        intWidth = X0.size(3)
        intHeight = X0.size(2)
        channel = X0.size(1)
        N = X0.size(0)
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
        self.flow = torch.zeros(N,intHeight,intWidth,2).cuda()
        self.mix_img = torch.zeros(N, 2*divnum, 3, intHeight,intWidth).cuda()
        #save_torchimg(X1[0].unsqueeze(0),'org')
        for i in range(N):

            self.flow[i] = self.get_flow(X0[i],X1[i],intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,intHeight,intWidth)[0]
        #print(full_flow.size())
        blur_out = torch.zeros(N,3,intHeight,intWidth).cuda()

        #vis.heatmap(self.flow[0,:,:,0])
        #vis.heatmap(self.flow[0,:,:,1])

        for i in range(N):
            #print(X1.size()) 
            x0input = X0[i].repeat(divnum,1,1,1)
            x1input = X1[i].repeat(divnum,1,1,1)
            #print(x1input.size()) 
            
            theta01 = torch.zeros(divnum,intHeight,intWidth,2).cuda()
            theta10 = torch.zeros(divnum,intHeight,intWidth,2).cuda()
            xl,yl  = torch.meshgrid(torch.Tensor(range(X0.size(2))),torch.Tensor(range(X0.size(3))))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            #vis.heatmap(self.flow[i][:,:,0])
            #vis.heatmap(self.flow[i][:,:,1])
            for d in range(divnum):
              
                theta_fw = (d+1) * self.flow[i]  / divnum 
                theta_bw = theta_fw - self.flow[i]
                #print(idflow.size(),theta_fw.size())  
                #vis.heatmap(self.flow[i][:,:,0])
                #vis.heatmap(self.flow[i][:,:,1])
                #vis.heatmap(idflow[i][0:,:,1])
                
                #print(theta_fw.size())
                theta01[d]  = (idflow - theta_fw).squeeze(0)
                theta10[d]  = (idflow - theta_bw).squeeze(0)
            #vis.heatmap(idflow[0,:,:,1])
            #vis.heatmap(theta01[15,:,:,0]) 
            theta01[:,:,:,0]=((theta01[:,:,:,0])-X0.size(3)/2)/(X0.size(3)/2)
            theta01[:,:,:,1]=((theta01[:,:,:,1])-X0.size(2)/2)/(X0.size(2)/2)
            #vis.heatmap(theta01[15,:,:,0])  
            #vis.heatmap(theta01[15,:,:,1])   
                  
            theta10[:,:,:,0]=((theta10[:,:,:,0])-X0.size(3)/2)/(X0.size(3)/2)
            theta10[:,:,:,1]=((theta10[:,:,:,1])-X0.size(2)/2)/(X0.size(2)/2)
            warped01 = F.grid_sample(input=x0input, grid=(theta01), mode='bilinear')
            warped10 = F.grid_sample(input=x1input, grid=(theta10), mode='bilinear')
            #print(warped10)  #数值为0-255
            
            #vis.image(warped10[0])
            mix_crop = torch.cat((warped01,warped10),0).clamp(0,255)
            #print(mix_crop.size())
            self.mix_img[i] = mix_crop
            blur_out[i] = mix_crop.sum(0)/(2*divnum)
        #print(blur_out[6])
        #save_torchimg(warped01[2].unsqueeze(0),'war')
        #save_torchimg(X1[6].unsqueeze(0),'org')
        save_torchimg(blur_out[0].unsqueeze(0),'blur')    
        
        return blur_out  , self.mix_img.clone().view(N,6*divnum,intHeight,intWidth) # N,3,255,255

    def forward(self,X0,X1,W,sx,sy):

        #print(flow[0][0])
      
        
       
        #print(sx.sum(1))
        
        W = W.unsqueeze(2)  
        divnum = W.size(1)//2
        map_sz = X0.size(2)
        #vis.heatmap(flow1[0][0])
        #vis.heatmap(flow1[0][1])

        
        #flow_avg = (flow1-flow2)/2

    
        #print(flow_loss)
        
        #print( X0.size(),W.size(),flow.size())
        #print(X0)
        #vis.image(X0[0])
        #vis.image(X1[0])
        out_blur = torch.zeros(W.size(0),3,map_sz,map_sz)
        for n in range(W.size(0)):
            xl,yl = torch.meshgrid(torch.Tensor(range(map_sz)),torch.Tensor(range(map_sz)))
            idflow = torch.stack((yl,xl)).unsqueeze(0).permute(0,2,3,1).cuda()
            theta01 = torch.zeros(divnum,map_sz,map_sz,2).cuda()
            theta10 = torch.zeros(divnum,map_sz,map_sz,2).cuda()
            img0rep = X0[n].unsqueeze(0).repeat(divnum,1,1,1)
            img1rep = X1[n].unsqueeze(0).repeat(divnum,1,1,1)

            '''
            for i in range(divnum):
                theta_f = (i+1) * self.flow[n].unsqueeze(0) / divnum   #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                #theta_b = (i+1) * flow2[n].unsqueeze(0) / divnum
                
                #input()
                theta_b = theta_f - flow1[n]
                theta01[i]  = (idflow - theta_f).squeeze(0)
                theta10[i]  = (idflow - theta_b).squeeze(0)
            '''
            for i in range(divnum):
                if flow_test:
                    theta_fw_x =    (sx[n,:i+1,:,:].sum(0))     * self.flow[n].unsqueeze(0)[:,:,:,0]     #这个是t  表示单位时间    光流x轴，向右运动为正，y轴向下运动为正
                    theta_fw_y =    (sy[n,:i+1,:,:].sum(0))     * self.flow[n].unsqueeze(0)[:,:,:,1] 
                    theta_fw = torch.stack((theta_fw_x,theta_fw_y)).permute(1,2,3,0)
                    #print(sx[n,:i+1,:,:].sum(0))
                    #input()
                    theta_bw = theta_fw - self.flow[n].unsqueeze(0)
                else:
                  
                    theta_fw =  (i+1) * self.flow[n] /divnum
                    #print(sx[n,:i+1,:,:].sum(0))
                    #input()
                    #print(theta_fw.size(),self.flow[n].size())
                    theta_bw = theta_fw - self.flow[n]

                #print(idflow.size(),theta_fw.size())     torch.clamp    ,0,frame_0.size(3)
                theta01[i]  = (idflow - theta_fw).squeeze(0)
                theta10[i]  = (idflow - theta_bw).squeeze(0)
            theta01[:,:,:,0]=((theta01[:,:,:,0])-map_sz/2)/(map_sz/2)
            theta01[:,:,:,1]=((theta01[:,:,:,1])-map_sz/2)/(map_sz/2)
            #vis.heatmap(theta01[-1,:,:,0])
                             
            theta10[:,:,:,0]=((theta10[:,:,:,0])-map_sz/2)/(map_sz/2)
            theta10[:,:,:,1]=((theta10[:,:,:,1])-map_sz/2)/(map_sz/2)
            #vis.heatmap(theta01[0,:,:,0])
            warped01 = F.grid_sample(input=img0rep, grid=(theta01.clamp(-1,1)), mode='bilinear').clamp(0,255)
            warped10 = F.grid_sample(input=img1rep, grid=(theta10.clamp(-1,1)), mode='bilinear').clamp(0,255)
            
            warp_mix = torch.cat((warped01,warped10),0)
            #print(warp_mix.size())
            if not W_test:
                W[n] = 1
            blur_1 = ((warp_mix * W[n]).sum(0)/(2*divnum)).clamp(0,255)
            #print(blur_1.size())
            #print(warped01.size(),warped01)
            #vis.image(warped01[5])
            out_blur[n] = blur_1
        
            
        #out_blur = (self.mix_img * W ).sum(1)/(2*self.divnum)
        #print(out_blur.size())
        save_torchimg(out_blur[0].unsqueeze(0),'out_blur')
        
        #save_torchimg(out_blur[2].unsqueeze(0),'out_blur2')
        #save_torchimg(out_blur[4].unsqueeze(0),'out_blur4')
        
        return out_blur  , 0
        

def get_advlabel():
    xx=12
    yy=12
    #xp = 10   
    #yp = 2
    adv_no = True
    #while adv_no:
    yp =  random.randint(0,24)
    xp =  random.randint(0,24)

            #if 1<xx+xp<23 and 1<yy+yp<23:
                #break 
    adv_cls = torch.zeros((1,5,25,25))
    adv_cls[:,:, yp,xp] = 1
    #adv_cls[:,:, yy+yp+1,xx+xp] = 1
    #adv_cls[:,:, yy+yp,xx+xp+1] = 1
    #adv_cls[:,:, yy+yp+1,xx+xp+1] = 1
    
    return adv_cls.long().cuda()
