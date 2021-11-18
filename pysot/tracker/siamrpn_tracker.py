# Copyright (c) SenseTime. All Rights Reserved.   .permute(1,2,0)
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
import cv2
from  visdom import Visdom

from PIL import Image , ImageDraw
vis=Visdom(env="tracker")

def save_onechannel(img,name,cod=[]):
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2XYZ)
    #print(img)
    #img=img.astype(np.uint8)
    img = img.unsqueeze(0)
    img = torch.nn.functional.interpolate(img , size=(255,255), mode='bilinear')
    #print(img[cod[0],cod[1],2],img[cod[0],cod[1],1],img[cod[0],cod[1],0])
    #img[cod[0],cod[1],2] = 20
    #img[cod[0],cod[1],1] = 20
    #img[cod[0],cod[1],0] = 200
    #img = Image.fromarray(img)
    #print(img.shape)
    #img = img.convert("RGB")
    #img = cv2.circle(img, (cod[0],cod[1]), 2, (0, 0, 255), 4)

    #draw = ImageDraw.Draw(img)
    r=1
    #draw.point((cod[1], cod[0]), fill=(0,0,255))
    img = img[0].permute(1,2,0).numpy().astype(np.uint8)
    #save_im.save('/workspace/ABA/img/{}.jpg'.format(name))
    #Image.open('/workspace/ABA/img/{}.jpg'.format(name))
    #print(img.shape)
    img = np.array(img)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    r=3
    draw.ellipse((cod[1]-r, cod[0]-r,cod[1]+r, cod[0]+r), fill=(255,0,0))
    img.save('/workspace/ABA/img/{}.jpg'.format(name))
    #cv2.imwrite('/workspace/ABA/img/{}.jpg'.format(name), save_im)
    '''
    img = Image.open('/workspace/ABA/img/{}.jpg'.format(name))
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)
    print(img_array.shape)
    img_array[cod[0],cod[1]] = [255, 0, 0]
    img3 = Image.fromarray(img_array)
    img3.save('/workspace/ABA/img/{}.jpg'.format(name))
    '''
    

class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def set_pos_size(self,pos,size):
        self.center_pos = pos
        self.size = size

    def vis_outmap(self,onemap=True):
        if onemap:
            vis.heatmap(self.oneori[0][0])
            vis.heatmap(self.onefin[0][0])
      
        else:
            for s in range(5):
                vis.heatmap(self.ori_map[0][s])
            for s in range(5):
                vis.heatmap(self.afterwin[0][s])

    def track(self, img, update=True, v=False, outputs_clean=None, x_c =None, v_onemap=False, saveimg = None):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        if x_c is not None:
            x_crop = x_c
        if outputs_clean is not None:
            outputs = outputs_clean
        else:
            outputs = self.model.track(x_crop)
        #print(outputs['cls'].size())
        
        
            
        score = self._convert_score(outputs['cls'])
        #print(score.shape)
        
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        if v:
            self.ori_map = score.copy().reshape(1,5,25,25) 
        if v_onemap:
          
            #maxid = np.argmax(score.reshape(5, -1),axis=1)
            bid = np.argmax(score)
            #ma = bid%625
            asd = bid//625
            #xx = ma%25
            #yy = ma//25
            #print(asd)
            self.oneori = score.copy().reshape(5,25,25)[asd].reshape(1,1,25,25)
            #maxid = maxid/625
            #print(maxid)

  

        
            
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        
        '''
        print(penalty.shape)
        pen = penalty.reshape((5,25,25))
        vis.heatmap(pen[0])
        vis.heatmap(pen[1])
        vis.heatmap(pen[2])
        vis.heatmap(pen[3])
        vis.heatmap(pen[4])
        input()
        '''
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        if v :
            
            bid = np.argmax(pscore)
            ma = bid%625
            asd = bid//625
            xx = ma%25
            yy = ma//25
            print(bid,asd,yy,xx)
            #vis.bar(score)
            sc_n = pscore.reshape((1,5,25,25))
            self.afterwin = sc_n 
            '''vis.heatmap(sc_n[0])
            vis.heatmap(sc_n[1])
            vis.heatmap(sc_n[2])
            vis.heatmap(sc_n[3])
            vis.heatmap(sc_n[4])'''
            #vis.heatmap(outputs['cls'][0,5,:,:])
            #vis.heatmap(outputs['cls'][0,6,:,:])
            #vis.heatmap(outputs['cls'][0,7,:,:])
            #vis.heatmap(outputs['cls'][0,8,:,:])
        if v_onemap:
            bid = np.argmax(pscore)
            ma = bid%625
            asd = bid//625
            self.onefin = pscore.copy().reshape(5,25,25)[asd].reshape((1,1,25,25))

        if saveimg is not None :
            bid = np.argmax(pscore)
            ma = bid%625
            asd = bid//625
            xx = ma%25
            yy = ma//25
            #print(bid,asd,yy,xx)
            #vis.bar(score)
            sc_n = score.reshape((5,1,25,25))
            #vis.heatmap(sc_n[asd][0]*255)
            iii = sc_n[asd].repeat(3,axis = 0)#.transpose(1,2,0)
            iii = torch.from_numpy(iii)
            #print(iii.size())
            
            
            save_onechannel(iii*255,saveimg,[int(yy*10.2),int(xx*10.2)])
            
        best_idx = np.argmax(pscore)
        #print(best_idx)
        #print(pscore.shape)
        
        '''if v:
            #vis.bar(pscore)
            ma = best_idx%625
            asd = best_idx//625
            xx = ma%25
            yy = ma//25
            #print(best_idx,asd,yy,xx)
            sc = pscore.reshape((5,25,25))
            vis.heatmap(sc[0])
            vis.heatmap(sc[1])
            vis.heatmap(sc[2])
            vis.heatmap(sc[3])
            vis.heatmap(sc[4])
            #vis.heatmap(sc_n[asd])'''
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        #print(lr)
        #input()
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        searbox = bbox.copy()
        searbox[0]-=searbox[2]/2
        searbox[1]-=searbox[3]/2
        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        #print(width,height)
        
        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
        #print()
        # udpate state
        if update:
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])
        copy_pos =  np.array([cx, cy]).copy()
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'copy_pos':copy_pos,
                'x_crop':x_crop,
                'bbox': bbox,
                'searbox':searbox,
                'best_score': best_score
               }
    
    def track_fixed(self, img,v=False):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
        if v :
            
            bid = np.argmax(score)
            ma = bid%625
            asd = bid//625
            xx = ma%25
            yy = ma//25
            print(bid,asd,yy,xx)
            #vis.bar(score)
            sc_n = score.reshape((1,5,25,25))
            '''vis.heatmap(sc_n[0])
            vis.heatmap(sc_n[1])
            vis.heatmap(sc_n[2])
            vis.heatmap(sc_n[3])
            vis.heatmap(sc_n[4])'''

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        # self.center_pos = np.array([cx, cy])
        # self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
