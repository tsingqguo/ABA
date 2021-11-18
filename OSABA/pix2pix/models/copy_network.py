import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn import functional as F




class cheng_UnetGenerator(nn.Module):
    def __init__(self,input_nc,output_nc,num_downs,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False):
        super(cheng_UnetGenerator,self).__init__()
        #降采样部分
        self.first_con=nn.Conv2d(input_nc,ngf,kernel_size=4,stride=2,padding=1,bias=False)
        self.down1=self.block_d(ngf,ngf*2,True)
        self.down2=self.block_d(ngf*2,ngf*4,True)
        self.down3=self.block_d(ngf*4,ngf*8,True)
        self.down4=self.block_d(ngf*8,ngf*8,True)
        self.midrelu1=nn.LeakyReLU(0.2,False)
        self.midcon=nn.Conv2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1,bias=False)
        self.midrelu2=nn.ReLU(False)

        #W上采样
        self.up1=nn.ConvTranspose2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1)
        self.up2=nn.BatchNorm2d(ngf*8)

        self.upw1=self.block_w(ngf*16,ngf*8,False)
        self.upw2=self.block_w(ngf*16,ngf*4,False)
        self.upw3=self.block_w(ngf*8,ngf*2,False)
        self.upw4=self.block_w(ngf*4,ngf,False)
        self.upw5=nn.ReLU(True)
        self.upw6=nn.ConvTranspose2d(ngf*2,output_nc,kernel_size=4,stride=2,padding=1)
        self.upw7=nn.Tanh()

        #对抗性光流拆分权重
        self.upf1=nn.ConvTranspose2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1)
        self.upf2=nn.BatchNorm2d(ngf*8)

        self.upwf1=self.block_w(ngf*16,ngf*8,False)
        self.upwf2=self.block_w(ngf*16,ngf*4,False)
        self.upwf3=self.block_w(ngf*8,ngf*2,False)
        self.upwf4=self.block_w(ngf*4,ngf,False)
        self.upwf5=nn.ReLU(True)
        self.upwf6=nn.ConvTranspose2d(ngf*2,34,kernel_size=4,stride=2,padding=1)
        self.upwf7=nn.Tanh()

        #光流上采样
        '''
        self.upf1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upf2 = nn.BatchNorm2d(ngf * 8)

        self.upf3 = self.block_f(ngf * 16 , ngf * 8)
        self.upf4 = self.block_f(ngf * 16 , ngf * 4)
        self.upf5 = self.block_f(ngf * 8 , ngf * 2)
        self.upf6 = self.block_f(ngf * 4 , ngf)
        self.upf7 = nn.LeakyReLU(0.2, False)
        self.upf8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upf9 = nn.Conv2d( ngf*2 ,ngf , kernel_size=3,stride=1, padding=1, bias= False)
        self.upf10 = nn.Conv2d( ngf ,ngf//2 , kernel_size=3,stride=1, padding=1, bias= False)
        self.upf11_1 = nn.Conv2d( ngf//2 , 2  , kernel_size=1,stride=1, padding=0, bias= False)
        self.upf11_2 = nn.Conv2d( ngf//2 , 2  , kernel_size=1,stride=1, padding=0, bias= False)
        self.upf12 = nn.LeakyReLU(0.2, False)
        '''
    def forward(self,x):
        #down
        x1=self.first_con(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x6=self.midrelu1(x5)
        x7=self.midcon(x6)
        x8=self.midrelu2(x7)

        #分支w
        x9=self.up1(x8)
        x10=self.up2(x9)
        #print(x10.size(),x5.size())  
        x10=torch.cat((x10,x5),1)
        x11=self.upw1(x10)
        x11=torch.cat((x11,x4),1)
        x12=self.upw2(x11)
        x12=torch.cat((x12,x3),1)
        x13=self.upw3(x12)
        x13=torch.cat((x13,x2),1)
        x14=self.upw4(x13)
        x15=self.upw5(x14)
        x15=torch.cat((x15,x1),1)
        x16=self.upw6(x15)
        x17=self.upw7(x16)

        #光流拆分
        xf9=self.upf1(x8)
        xf10=self.upf2(xf9)
        #print(x10.size(),x5.size())  
        xf10=torch.cat((xf10,x5),1)
        xf11=self.upwf1(xf10)
        xf11=torch.cat((xf11,x4),1)
        xf12=self.upwf2(xf11)
        xf12=torch.cat((xf12,x3),1)
        xf13=self.upwf3(xf12)
        xf13=torch.cat((xf13,x2),1)
        xf14=self.upwf4(xf13)
        xf15=self.upwf5(xf14)
        xf15=torch.cat((xf15,x1),1)
        xf16=self.upwf6(xf15)
        xf17=self.upwf7(xf16)
        sx = xf17[:,0:17,:,:]
        sy = xf17[:,17:34,:,:]
        sx = F.softmax(sx,1)
        sy = F.softmax(sy,1)

        return sx,sy, x17

    def block_d(self,inc,ouc,use_bias):
        downconv = nn.Conv2d(inc , ouc , kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        morm = nn.BatchNorm2d(ouc)

        block = [downrelu,downconv,morm]

        return nn.Sequential(*block)

    def block_w(self,inc,ouc,use_bias):
        upconv = nn.ConvTranspose2d(inc,ouc,kernel_size=4, stride=2,padding=1)
        uprelu = nn.ReLU(False)
        morm = nn.BatchNorm2d(ouc)

        block = [uprelu ,upconv ,morm]
        return nn.Sequential(*block)

    def block_f(self,inc,ouc):
        upsize = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        upconv1 = nn.Conv2d( inc,ouc , kernel_size=3,stride=1, padding=1)
        upconv2 = nn.Conv2d( ouc,ouc , kernel_size=3,stride=1, padding=1)
        uprelu = nn.LeakyReLU(inplace=True)
        morm = nn.BatchNorm2d(ouc)

        block = [uprelu ,upsize ,upconv1, morm ,uprelu, upconv2,morm]
        return nn.Sequential(*block)