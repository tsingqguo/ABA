from OSABA.pix2pix.options.test_options0 import TestOptions
from OSABA.pix2pix.models import create_model
import os

opt = TestOptions().parse()

# modify some config
'''Attack Search Regions'''
opt.model = 'G_search_L2_500' # only cooling
# opt.model = 'G_search_L2_500_regress' # cooling + shrinking
opt.netG = 'unet_256'   # cheng_unet_256   unet_256

# create and initialize model
'''create perturbation generator'''
GAN = create_model(opt)  # create a model given opt.model and other options
GAN.load_path = 'osaba_net_G.pth'
GAN.setup(opt)  # # regular setup: load and print networks; create schedulers
GAN.eval()
