from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import torch.nn as nn
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.tracker.tracker_builder_cross import build_trackerC
from pysot.models.model_builder_cross import CROSSModelBuilder
from attacker import attacker
from attacker_dimp import attacker as attacker_dimp
from attacker_kys import attacker as attacker_kys
from onestepblur.model_blur import blur_onestep
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import rc
rc("pdf", fonttype=42)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
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
test_mode = 'testall'   # figall  debug    testall  visionall  transfer   onestep   compare  transfer_tracker  onestep_online
attack_mode = 'guo_speed'    #     guo  nor_blur    whitebox     guo_speed   speed_blackbox  
attack_bool = True
tran_path = '/workspace/ABA/experiments/siamrpn_mobilev2_l234_dwxcorr/'    #这个是产生攻击样本的tracker    siamrpn_mobilev2_l234_dwxcorr
tran_tracker_size = 224
onestep_path = '/workspace/ABA/onestepblur/save/check_guo_e52.pth'

def get_move_dis(box0,box1):
    cx0 = box0[0]+0.5*box0[2]
    cy0 = box0[1]+0.5*box0[3]
    cx1 = box1[0]+0.5*box1[2]
    cy1 = box1[1]+0.5*box1[3]
    d_mov = ((cx0-cx1)**2+(cy0-cy1)**2)** 0.5
    return d_mov

def save_torchimg(img,name):
    save_im = img[0].detach().permute(1,2,0).cpu().numpy()
    cv2.imwrite('/workspace/ABA/compare_img/{}.png'.format(name), save_im)

def fig_bbox(ter,pred_bbox,name):
    npimg = ter[0].detach().permute(1,2,0).cpu().numpy()
    #print(pred_bbox)
    
    pred_bbox[0]+= 128
    pred_bbox[1]+= 128
    pred_bbox = list(map(int, pred_bbox))
    cv2.rectangle(npimg, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 0, 255), 1)
    cv2.imwrite('/workspace/ABA/demo/{}.jpg'.format(name),npimg )

def save_pdf(ter,pred_bbox,name):
    npimg = ter[0].detach().permute(1,2,0).cpu().numpy().astype(np.uint8)
    pred_bbox[0]+= 128
    pred_bbox[1]+= 128
    pred_bbox = list(map(int, pred_bbox))
    print(npimg)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    rect = plt.Rectangle((pred_bbox[0],pred_bbox[1]),pred_bbox[2],pred_bbox[3])
    ax.add_patch(rect)
    #plt.plot(x,y)
    plt.imshow(npimg)
    
    plt.savefig('/workspace/ABA/img/{}.pdf'.format(name)  ,dpi=500)   #

def main():
    # load config
    cfgC = cfg.clone()
    cfg.merge_from_file(args.config)
    vis_img = False
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()
 
    #load_pretrain(blur_attack, onestep_path)
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    
    '''
    for name,parameters in model.named_parameters():
            print(name,':',parameters.size())
    '''
    # build tracker
    tracker = build_tracker(model)
    #tracker_att = build_tracker(model)
    if test_mode == 'transfer'  :

        cfg.merge_from_file(tran_path + 'config.yaml')
        tran_model = ModelBuilder()
        tran_model = load_pretrain(tran_model, tran_path + 'model.pth').cuda().train()
        tran_tracker = build_tracker(tran_model)
    if test_mode == 'transfer_tracker':   
        cfg.merge_from_file(tran_path + 'config.yaml')
        tran_tracker = build_tracker(model) 
        if 'kys' in tran_path:
            ack_source = attacker_kys()
        elif 'dimp' in tran_path:
            ack_source = attacker_dimp()
        else:
            sb
    ack = attacker()
    

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = 'opaba' #args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019'] :#and not os.path.exists(os.path.join('attack', args.dataset, model_name)):  #,'baseline'
        
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            #if os.path.exists(os.path.join('/workspace/ABA/experiments/siamrpn_r50_l234_dwxcorr/results/VOT2018/normblur/baseline',video.name)):
                #continue
            #print(video.name,os.path.join('result', args.dataset, model_name,'baseline',video.name))
            
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            print('Processing video:', video.name, model_name)
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
            
                    tracker.init(img, gt_bbox_)
                    if test_mode == 'transfer' or test_mode == 'transfer_tracker':
                        tran_tracker.init(img, gt_bbox_)
                    # tracker_att.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    X0 = img.copy()
                elif idx > frame_counter:
                    #tracker_att.track(img)
                    if idx %2 == 0 and attack_bool and test_mode == 'testall':
                        
                        #vis.image(img.transpose(2,0,1))
                        
                        X1=img.copy()
                        
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True)
                        #print(img_attack.size())
                        #outputs = tracker.track(out_ack['norm_org'])
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'])
                        
                        
                        save_torchimg(out_ack['adv_x_crop'],'vottest')
                        
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)
                        input()
                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer':
                        X1=img.copy()
                        
                    
                        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        s_z = np.sqrt(w_z * h_z)
                        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                        x0_crop = tracker.get_subwindow(X0, tracker.center_pos,
                                                    cfg.TRACK.INSTANCE_SIZE,
                                                    round(s_x), tracker.channel_average)
                        x1_crop = tracker.get_subwindow(X1, tracker.center_pos,
                                                    cfg.TRACK.INSTANCE_SIZE,
                                                    round(s_x), tracker.channel_average) 

                        out_ack = ack(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tran_tracker, speed= True ,crop = [x0_crop,x1_crop])
                        #tran_tracker.set_pos_size(tracker.center_pos , tracker.size)
                        #out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tran_tracker)
                        #print(out_ack['adv_x_crop'].size())
                        rs_img = nn.Upsample(size=cfg.TRACK.INSTANCE_SIZE, mode='bilinear')(out_ack['adv_x_crop'])
                        #print(rs_img.size())
                        #save_torchimg(rs_img,'vottest')
                        outputs = tracker.track(img, x_c = rs_img)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()
                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer_tracker':
                        X1=img.copy()
                        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        s_z = np.sqrt(w_z * h_z)
                        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                        x0_crop = tracker.get_subwindow(X0, tracker.center_pos,
                                                    tran_tracker_size,
                                                    round(s_x), tracker.channel_average)
                        x1_crop = tracker.get_subwindow(X1, tracker.center_pos,
                                                    tran_tracker_size,
                                                    round(s_x), tracker.channel_average) 

                        out_ack = ack_source(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tran_tracker, speed= True ,crop = [x0_crop,x1_crop])
                        #tran_tracker.set_pos_size(tracker.center_pos , tracker.size)
                        #out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tran_tracker)
                        #print(out_ack['adv_x_crop'].size())
                        #rs_img = nn.Upsample(size=cfg.TRACK.INSTANCE_SIZE, mode='bilinear')(out_ack['adv_x_crop'])
                        #print(rs_img.size())
                        #save_torchimg(rs_img,'vottest')
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'])
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)
                        input()
                    
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
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
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
    else: #if  not os.path.exists(os.path.join('results', args.dataset, model_name)):
        # OPE tracking
        frame_num = 0
        frame_attack = 0
        attack_time = 0

        time_start_all = time.time()
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            #if os.path.exists(os.path.join('/workspace/ABA/experiments/siamrpn_mobilev2_l234_dwxcorr/results/LaSOT/guo_speed_m2lasot','{}.txt'.format(video.name))):
                #print('pass   ' +video.name )
                #continue
                #print(os.path.join('/workspace/ABA/experiments/siamrpn_mobilev2_l234_dwxcorr/results/LaSOT/guo_speed_m2lasot','{}.txt'.format(video.name)))
            tq=0
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            d_m = 0
            box1 =None
            print('Processing video:', video.name, model_name)
            for idx, (img, gt_bbox) in enumerate(video):
                frame_num+=1
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    if test_mode == 'transfer' or test_mode == 'transfer_tracker':
                        tran_tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                    box0 = pred_bbox
                    X0 = img.copy()
                    stop=False
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
                        #aaa = input()
                        #print(aaa)
                        frame_attack+=1
                        X1=img.copy()
                        att_start = time.time()
                        #print(np.mean(img))
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True)
                        att_end = time.time()
                        attack_time += att_end-att_start
                        #print(img_attack.size())
                        #outputs = tracker.track(out_ack['norm_org'])
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'],v=0)
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(out_ack['warp'][0].size())
                        #save_pdf(out_ack['adv_x_crop'],outputs['searbox'],'adv_x_crop')

                        
                        print('after attacking ',overlap)
                        aaa =input()
                        
                    elif ( idx %5 == 0 or idx %5 == 1 ) and idx != 10 and attack_bool and test_mode == 'figall':
                        
                        #X0 = im0
                        #aaa = input()
                        #print(aaa)
                        print('adv', idx )
                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True)
                        #print(img_attack.size())
                        
                        #temp_pos = output_clean['copy_pos']
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'],update=False,saveimg = 'response_blur')
                        #print(out_ack['normblur_crop'].size())
                        output_clean = tracker.track(img,update=False, saveimg = 'response_clean')
                        output_norm = tracker.track(img, x_c = out_ack['normblur_crop'],saveimg = 'response_blur')
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(out_ack['warp'][0].size())
                        '''
                        if idx>=30 :
                            print('in',outputs['searbox'])
                            tq += 18
                            outputs['searbox'][0]+=0.2 *tq
                            outputs['searbox'][1]+=0.745 *tq
                            fig_bbox(out_ack['adv_x_crop'],outputs['searbox'],'adv/demo_{}'.format(idx))
                        else:
                            fig_bbox(out_ack['adv_x_crop'],outputs['searbox'],'adv/demo_{}'.format(idx))
                        '''
                        fig_bbox(out_ack['normblur_crop'],output_norm['searbox'],'norm/demo_{}'.format(idx))
                        #fig_bbox(output_clean['x_crop'],output_clean['searbox'],'clean/demo_{}'.format(idx))
                        X0 = img.copy()
                        #save_torchimg(out_ack['frame'][0],'frame0')
                        #save_torchimg(out_ack['frame'][1],'frame1')
                        #save_torchimg(out_ack['adv_x_crop'],'adv_x_crop')
                        print(overlap)
                        
                        if overlap <0.2:
                            stop = True
                        '''
                        aaa =input()
                        if aaa == '111':
                            for i in range(out_ack['warp'][0].size(0)):
                                save_torchimg(out_ack['warp'][0],'warp0_{}'.format(i))
                                save_torchimg(out_ack['warp'][1],'warp1_{}'.format(i))
                            dg
                        '''
                    elif idx %5 == 0 and attack_bool and test_mode == 'saveimg':
                        
                        #X0 = im0
                        #aaa = input()
                        #print(aaa)
                       
                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tracker, speed = True)
                        #print(img_attack.size())
                        output_clean = tracker.track(img,update=False, saveimg = 'response_clean')
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'],saveimg = 'response_blur')
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(out_ack['warp'][0].size())
                      
                        
                        #save_torchimg(out_ack['frame'][0],'frame0')
                        save_torchimg(out_ack['frame'][1],'frame1_{}_{}'.format(video.name,idx))
                        save_torchimg(out_ack['adv_x_crop'],'adv_x_crop_{}_{}'.format(video.name,idx))
                        #print(overlap)
                        #aaa =input()
                        
                    elif idx %5 == 0 and attack_bool and test_mode == 'compare':
                        
                        #X0 = im0
                        
                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = 'guo' ,model_attacked = tracker)
                        #print(img_attack.size())
                        
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'],v=0, update=False)
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)

                        out_ack = ack(img0=X0,img1=X1,mode = 'guo_speed' ,model_attacked = tracker,speed = False)
                        #print(img_attack.size())
                        
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'],v=0, update=False)
                        
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)

                        out_ack = ack(img0=X0,img1=X1,mode = 'guo_speed' ,model_attacked = tracker,speed = True)
                        #print(img_attack.size())
                        
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'],v=0)
                        
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)
                        input()
                    elif idx %5 == 0 and attack_bool and test_mode == 'onestep':
                        X1=img.copy()
                        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        s_z = np.sqrt(w_z * h_z)
                        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                        x0 = tracker.get_subwindow(X0, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
                        x1 = tracker.get_subwindow(X1, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
                        #out_ack = ack(img0=X0,img1=X1,mode = 'guo_speed' ,model_attacked = tracker,speed = False)               
                        img_mix = torch.stack((x0,x1)).view(1,6,255,255)
                        blur_one ,  w_out1 ,  w_out2  = blur_attack(img_mix)    #,out_ack['flow_full_crop']
                        outputs = tracker.track(img, x_c = blur_one,v=0)
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()
                   
                    elif idx %5 == 0 and attack_bool and test_mode == 'onestep_online':
                        X1=img.copy()
                        out_ack = ack(img0=X0,img1=X1,mode = 'guo_speed' ,model_attacked = tracker, speed = True)
                        blur_im = ack.train_onestep(img0=X0,img1=X1 ,tracker = tracker,blur_attack=blur_attack,attacker_out=out_ack )
                        #print(img_attack.size())
                        
                        outputs = tracker.track(img, x_c = blur_im,v=0)
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        print(overlap)
                        input()
                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer':
                        X1=img.copy()
                        
                        tran_tracker.set_pos_size(tracker.center_pos , tracker.size)
                        out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tran_tracker)
                        #print(img_attack.size())
                        
                        outputs = tracker.track(img, x_c = out_ack['adv_x_crop'])
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        #overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()    
                    elif idx %5 == 0 and attack_bool and test_mode == 'transfer_tracker':
                        X1=img.copy()
                        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        s_z = np.sqrt(w_z * h_z)
                        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                        x0_crop = tracker.get_subwindow(X0, tracker.center_pos,
                                                    tran_tracker_size,
                                                    round(s_x), tracker.channel_average)
                        x1_crop = tracker.get_subwindow(X1, tracker.center_pos,
                                                    tran_tracker_size,
                                                    round(s_x), tracker.channel_average) 

                        out_ack = ack_source(img0=X0,img1=X1,mode = 'guo_crop_speed' , model_attacked = tran_tracker, speed= True ,crop = [x0_crop,x1_crop])
                        #tran_tracker.set_pos_size(tracker.center_pos , tracker.size)
                        #out_ack = ack(img0=X0,img1=X1,mode = attack_mode ,model_attacked = tran_tracker)
                        #print(out_ack['adv_x_crop'].size())
                        rs_img = nn.Upsample(size=cfg.TRACK.INSTANCE_SIZE, mode='bilinear')(out_ack['adv_x_crop'])
                        #print(rs_img.size())
                        #save_torchimg(rs_img,'vottest')
                        outputs = tracker.track(img, x_c = rs_img)
                    
                        #overlap_clean = vot_overlap(outputs1['bbox'], gt_bbox, (img.shape[1], img.shape[0]))

                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()
                    elif idx %5 == 0 and test_mode == 'visionall' and attack_bool:
                        
                        
                        
                        X1=img.copy()
                        out_norm = ack(X0,X1,mode = 'guo' ,model_attacked = tracker,normblur = True)
                        #print(out_ack['x_crop_normblur'].size())
                        #vis.image(img.transpose(2,0,1))
                        #print(out_norm['norm_org'])
                    
                        pred_bbox_normblur = tracker.track(out_norm['norm_org'] ,   update=False)['bbox']  #没有调整的模糊

                        #out_norm = ack(X0,X1,mode = 'guo' ,model_attacked = tracker,normblur = True)
                        #pred_bbox_normblur = tracker.track(out_norm['norm_org'] ,   update=False)['bbox']

                        out_ack = ack(X0,X1,mode = 'guo_speed' ,model_attacked = tracker)
                        pred_bbox_advblur  = tracker.track(img, x_c = out_ack['adv_x_crop']     , update=False)['bbox']

                        w_z = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        h_z = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(tracker.size)
                        s_z = np.sqrt(w_z * h_z)
                        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
                        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                        X0 = tracker.get_subwindow(X0, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
                        X1 = tracker.get_subwindow(X1, tracker.center_pos,
                                       cfg.TRACK.INSTANCE_SIZE,
                                       round(s_x), tracker.channel_average)
                        img_mix = torch.stack((X0,X1)).view(1,6,255,255)
                        blur_one ,  w_out1 ,  w_out2  = blur_attack(img_mix)
                        onestep_org = tracker.get_orgimg(img.copy(), blur_one, tracker.center_pos,cfg.TRACK.INSTANCE_SIZE,round(s_x), tracker.channel_average).copy()
                        pred_bbox_one = tracker.track(img, x_c = blur_one, update=False ,v=0)['bbox']   # onestep 模糊


                        #out_ack = ack(X0,X1,mode = 'onestep' ,model_attacked = tracker)
                        pred_bbox   = tracker.track(img,  v =False ,update=True)['bbox']    #干净图像
                        overlap_adv= vot_overlap(pred_bbox, pred_bbox_advblur, (img.shape[1], img.shape[0]))
                        print(overlap_adv)
                        
                        if overlap_adv < 1 :
                            gt_bbox = list(map(int, gt_bbox))
                            pred_bbox = list(map(int, pred_bbox))
                            pred_bbox_advblur  = list(map(int, pred_bbox_advblur))
                            pred_bbox_normblur  = list(map(int, pred_bbox_normblur))
                            pred_bbox_one  = list(map(int, pred_bbox_one))

                            cv2.putText(img, 'clean', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(out_norm['norm_org'], 'normblur', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(out_ack['advblur_org'], 'adv_speed', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(onestep_org, 'onestep', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 1)
                            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 1)
                            cv2.rectangle(out_norm['norm_org'], (pred_bbox_normblur[0], pred_bbox_normblur[1]),
                                  (pred_bbox_normblur[0]+pred_bbox_normblur[2], pred_bbox_normblur[1]+pred_bbox_normblur[3]), (0, 255, 255), 1)  
                            cv2.rectangle(out_ack['advblur_org'], (pred_bbox_advblur[0], pred_bbox_advblur[1]),
                                  (pred_bbox_advblur[0]+pred_bbox_advblur[2], pred_bbox_advblur[1]+pred_bbox_advblur[3]), (0, 255, 255), 1)   
                            cv2.rectangle(onestep_org, (pred_bbox_one[0], pred_bbox_one[1]),
                                  (pred_bbox_one[0]+pred_bbox_one[2], pred_bbox_one[1]+pred_bbox_one[3]), (0, 255, 255), 1)   

                            cv2.imwrite('/workspace/ABA/clean.jpg', img)
                            cv2.imwrite('/workspace/ABA/norm.jpg', out_norm['norm_org'])
                            cv2.imwrite('/workspace/ABA/adv.jpg', out_ack['advblur_org'])
                            cv2.imwrite('/workspace/ABA/one.jpg', onestep_org)

                            input()
                            #vis_img =True
                    else: 
                        #X0 = X1.copy()
                        #print('clean', idx , tracker.center_pos)
                        X0 = img.copy()
                        outputs = tracker.track(img)
                        '''
                        if idx>=30:
                            tq+=28
                            outputs['searbox'][0]+=0.3 *tq
                            outputs['searbox'][1]+=0.545 *tq
                        '''
                        #fig_bbox(outputs['x_crop'],outputs['searbox'],'norm/demo_{}'.format(idx))
                        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
                        #print(overlap)
                        #input()
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
            print('attack frames are ' , frame_attack )
            print('attack time is  ', attack_time )
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
        time_end_all = time.time()
        print('#############################')
        print('totally time cost', time_end_all - time_start_all)
        print('all frames are ' , frame_num )
        print('attack frames are ' , frame_attack )
        print('attack time is  ', attack_time )

if __name__ == '__main__':
    main()
