import torch
from .base_model import BaseModel
from . import networks

'''siamrpn++'''
from siamRPNPP import SiamRPNPP
from data_utils import normalize

'''hyper-parameters, which may need to be tuned'''
cls_thres = 0.7

class GtemplateL21000regressModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    '''Only use G, and train it with L1 Loss & fooling loss'''
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
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=1000, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'cls', 'reg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['template_clean1','template_adv1']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight_L2 = 1000
            self.init_weight_cls = 0.1
            self.init_weight_reg = 1
            self.cls_margin = -5
            self.side_margin1 = -5
            self.side_margin2 = -5
            self.weight_L2 = self.init_weight_L2
            self.weight_cls = self.init_weight_cls
            self.weight_reg = self.init_weight_reg
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        '''siamrpn++'''
        self.siam = SiamRPNPP()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.template_clean255 = input[0].squeeze(0).cuda() # pytorch tensor, shape=(1,3,127,127) [0,255]
        self.template_clean1 = normalize(self.template_clean255)
        # print('clean image shape:',self.init_frame_clean.size())
        self.X_crops = input[1].squeeze(0).cuda() # pytorch tensor, shape=(N,3,255,255)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''pad to (128,128)'''
        template128_clean = torch.zeros(1,3,128,128).cuda()
        template128_clean[:,:,1:,1:] = self.template_clean1

        template128_adv = template128_clean + self.netG(template128_clean)  # Residual form: G(A)+A
        '''Then crop back to (127,127)'''
        self.template_adv1 = template128_adv[:,:,1:,1:]
        self.template_adv255 = self.template_adv1 * 127.5 + 127.5 # [0,255]

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.template_adv1, self.template_clean1) * self.weight_L2
        attention_mask = (self.score_maps_clean > cls_thres)
        num_attention = int(torch.sum(attention_mask))
        if num_attention > 0:
            # label = torch.zeros((num_attention,),dtype=torch.long).cuda()

            score_map_adv_att = self.score_maps_adv[attention_mask]
            reg_adv_att = self.reg_res_adv[2:4,attention_mask]
            # score_map_adv_att = self.score_maps_clean[self.score_maps_clean > cls_thres]
            # self.loss_MCls = self.criterionMCls(score_map_adv_att, label) * weight
            '''original loss(2019.9.8)'''
            # self.loss_fool = torch.mean(score_map_adv_att[:, 1] - score_map_adv_att[:, 0]) * self.weight
            '''new loss(hinge loss)(2019.9.9)'''
            self.loss_cls = torch.mean(torch.clamp(score_map_adv_att[:, 1] - score_map_adv_att[:, 0], min=self.cls_margin)) * self.weight_cls
            self.loss_reg = (torch.mean(torch.clamp(reg_adv_att[0,:],min=self.side_margin1))+
                             torch.mean(torch.clamp(reg_adv_att[1,:],min=self.side_margin2))) * self.weight_reg
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L2 + self.loss_cls + self.loss_reg
        else:
            self.loss_G = self.loss_G_L2
        # self.loss_G = self.loss_fool
        self.loss_G.backward()

    def optimize_parameters(self):
        '''One forward & backward pass. One update of parameters'''
        '''forward pass'''
        # 1. predict with clean template
        with torch.no_grad():
            self.siam.model.template(self.template_clean255)
            self.score_maps_clean = self.siam.get_heat_map(self.X_crops,softmax=True)#(5HWN,),with softmax
        # 2. adversarial attack with GAN
        self.forward()  # compute fake image
        # 3. predict with adversarial template
        self.siam.model.template(self.template_adv255)
        self.score_maps_adv, self.reg_res_adv = self.siam.get_cls_reg(self.X_crops,softmax=False)#(5HWN,2)without softmax,(5HWN,4)
        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights