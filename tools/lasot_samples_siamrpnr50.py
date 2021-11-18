import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import os
import pdb
from tqdm import tqdm
import argparse
from attacker import attacker
from pysot.utils.model_load import load_pretrain
from pysot.tracker.tracker_builder import build_tracker
from pysot.models.model_builder import ModelBuilder
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect #, overlap_ratio
#from pysot.utils.utils1 import get_axis_aligned_rect
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--lasotn', type=str,
        help='lasotn')
args = parser.parse_args()

feature_path ='/workspace/ABA/defence_data'
cfg.merge_from_file('/cheng/pysot-deepmix-qikan/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
model = ModelBuilder()
model = load_pretrain(model, '/cheng/pysot-deepmix-qikan/experiments/siamrpn_r50_l234_dwxcorr/model.pth').cuda().eval()
tracker = build_tracker(model)
ack = attacker()

reset = 1; frame_max = 300


video_path = '/dataset/lasot/LaSOTBenchmark'
lists = open('/workspace/ABA/defence_data/'+'lasot{}'.format(args.lasotn)+'.txt','r')
list_file = [line.strip() for line in lists]
category = os.listdir(video_path)
category.sort()


#print(category)

for tmp_cat in category:
    videos = os.listdir(join(video_path, tmp_cat)); videos.sort()    
    for video in videos:
        template_z = []
        search_clean = []
        search_adv = []

        if video not in list_file:
            continue
        print('process  ' , video)        
        gt_path = join(video_path,tmp_cat,video, 'groundtruth.txt')
        ground_truth = np.loadtxt(gt_path, delimiter=',')
        num_frames = len(ground_truth);  #num_frames = min(num_frames, frame_max)
        img_path = join(video_path,tmp_cat,video, 'img')
        imgFiles = [join(img_path,'%08d.jpg') % i for i in range(1,num_frames)]
        #imgFiles1 = [join(img_path,'%08d.jpg') % i for i in range(2,num_frames+1)]
        #frame = 0
        #frame_z = 0
        for frame_z in tqdm(range(0 ,num_frames-51,200)): #生成模板循环
            Polygon = ground_truth[frame_z]
            cx, cy, w, h = get_axis_aligned_bbox(Polygon)
            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            if w*h!=0:
                image_file = imgFiles[frame_z]
                
                target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                im = cv2.imread(image_file)  # HxWxC
                tracker.init(im,gt_bbox_)  # init tracker
                zf = torch.cat(tracker.model.zf).detach().cpu().numpy()
                #print(zf[0].size(),zf[1].size(),zf[2].size())    # 3 * [1, 256, 7, 7]
                #print(zf.size())
                
                
                for frame in (range(frame_z + 1  ,min(num_frames-5 ,frame_z+50),1)):
                    
                    if frame%5==0:
                        image_file0 = imgFiles[frame]
                        image_file1 = imgFiles[frame+1]
                        if not image_file0 or not image_file1:
                            break
                        im0 = cv2.imread(image_file0)  # HxWxC
                        im1 = cv2.imread(image_file1)
                        out_ack = ack(img0=im0,img1=im1 , mode = 'guo_speed' ,model_attacked = tracker, speed = True)
                        xf_adv = torch.cat(out_ack['xf_adv']).detach().cpu().numpy()                
                        xf_clean = torch.cat(out_ack['xf_clean']).detach().cpu().numpy()
                        
                        #print(xf_adv.size(),xf_clean.size())
                        #outputs = tracker.track(im1, outputs_clean = out_ack['outputs_clean'])
                        #overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        template_z.append(zf)
                        search_clean.append(xf_clean)
                        search_adv.append(xf_adv)


                    else: #正常跟踪获得偏移
                        image_file0 = imgFiles[frame]
                        img = cv2.imread(image_file0)
                        outputs = tracker.track(img)    
                    #frame = frame + 1      
            #frame_z = frame_z + 5 #skip

        template_z=np.concatenate(template_z) 
        print(template_z.shape)
        fgsedg
        search_clean=np.concatenate(search_clean) 
        search_adv=np.concatenate(search_adv) 
        np.save(feature_path+'/zf/'+video,template_z)
        np.save(feature_path+'/xf_c/'+video,search_clean)
        np.save(feature_path+'/xf_adv/'+video,search_adv)
        print('save   ', video ,'samples numbers are  ' , len(template_z))
        #input()


