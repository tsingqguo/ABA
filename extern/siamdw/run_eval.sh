python ./siamese_tracking/test_siamfc.py --arch SiamFCRes22 --resume ./snapshot/CIResNet22.pth --dataset OTB2015
python ./siamese_tracking/test_siamfc.py --arch SiamFCNext22 --resume ./snapshot/CIRNext22.pth --dataset OTB2015
python ./siamese_tracking/test_siamfc.py --arch SiamFCIncep22 --resume ./snapshot/CIRIncep22.pth --dataset OTB2015
python ./lib/core/eval_otb.py OTB2015 ./result SiamFC* 0 1