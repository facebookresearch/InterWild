# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import math
import random
from glob import glob
from pycocotools.coco import COCO
from config import cfg
from utils.mano import mano
from utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, get_iou, load_ply
from utils.vis import vis_keypoints, save_obj

class HIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        assert data_split == 'test', 'only testing is supported for HIC dataset'
        self.data_path = osp.join('..', 'data', 'HIC', 'data')
        self.focal = (525.0, 525.0)
        self.princpt = (319.5, 239.5)

        # HIC joint set
        self.joint_set = {
                        'joint_num': 28, 
                        'joints_name': ('R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Thumb_2', 'R_Thumb_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Thumb_2', 'L_Thumb_3'),
                        'flip_pairs': [ (i,i+14) for i in range(14)]
                        }
        self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.data_path, 'HIC.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.data_path, img['file_name'])
            hand_type = ann['hand_type']

            # bbox
            body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
            if body_bbox is None:
                continue

            # mano mesh
            if ann['right_mano_path'] is not None:
                right_mano_path = osp.join(self.data_path, ann['right_mano_path'])
            else:
                right_mano_path = None
            if ann['left_mano_path'] is not None:
                left_mano_path = osp.join(self.data_path, ann['left_mano_path'])
            else:
                left_mano_path = None

            datalist.append({
                'aid': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'body_bbox': body_bbox,
                'hand_type': hand_type,
                'right_mano_path': right_mano_path,
                'left_mano_path': left_mano_path})
            
        return datalist
    

    def get_bbox_from_mesh(self, mesh):
        x = mesh[:,0] / mesh[:,2] * self.focal[0] + self.princpt[0]
        y = mesh[:,1] / mesh[:,2] * self.focal[1] + self.princpt[1]
        xy = np.stack((x,y),1)
        bbox = get_bbox(xy, np.ones_like(x))
        return bbox

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, body_bbox = data['img_path'], data['img_shape'], data['body_bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # mano coordinates
        right_mano_path = data['right_mano_path']
        if right_mano_path is not None:
            rhand_mesh = load_ply(right_mano_path)
        else:
            rhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        left_mano_path = data['left_mano_path']
        if left_mano_path is not None:
            lhand_mesh = load_ply(left_mano_path)
        else:
            lhand_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        mano_mesh_cam = np.concatenate((rhand_mesh, lhand_mesh))
        
        inputs = {'img': img}
        targets = {'mano_mesh_cam': mano_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'mpvpe_sh': [None for _ in range(sample_num)],
                    'mpvpe_ih': [None for _ in range(sample_num*2)],
                    'rrve': [None for _ in range(sample_num)],
                    'mrrpe': [None for _ in range(sample_num)],
                    'bbox_iou': [None for _ in range(sample_num*2)]
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            out = outs[n]
            mesh_out = np.concatenate((out['rmano_mesh_cam'], out['lmano_mesh_cam'])) * 1000 # meter to milimeter
            mesh_gt = out['mano_mesh_cam_target'] * 1000 # meter to milimeter
            
            # visualize
            vis = False
            if vis:
                filename = str(annot['aid'])
                img = out['img'].transpose(1,2,0)[:,:,::-1]*255
            
                lhand_bbox = out['lhand_bbox'].reshape(2,2).copy()
                lhand_bbox[:,0] = lhand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                lhand_bbox[:,1] = lhand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                lhand_bbox = lhand_bbox.reshape(4)
                img = cv2.rectangle(img.copy(), (int(lhand_bbox[0]), int(lhand_bbox[1])), (int(lhand_bbox[2]), int(lhand_bbox[3])), (255,0,0), 3)
                rhand_bbox = out['rhand_bbox'].reshape(2,2).copy()
                rhand_bbox[:,0] = rhand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                rhand_bbox[:,1] = rhand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                rhand_bbox = rhand_bbox.reshape(4)
                img = cv2.rectangle(img.copy(), (int(rhand_bbox[0]), int(rhand_bbox[1])), (int(rhand_bbox[2]), int(rhand_bbox[3])), (0,0,255), 3)
                cv2.imwrite(filename + '.jpg', img)

                save_obj(out['rmano_mesh_cam'], mano.face['right'], filename + '_right.obj')
                save_obj(out['lmano_mesh_cam'] + out['rel_trans'].reshape(1,3), mano.face['left'], filename + '_left.obj')
 
            # bbox IoU
            bb2img_trans = out['bb2img_trans']
            for idx, h in enumerate(('right', 'left')):
                if annot[h + '_mano_path'] is None:
                    continue
                bbox_out = out[h[0] + 'hand_bbox'] # xyxy in cfg.input_body_shape space
                if h == 'right':
                    bbox_gt = self.get_bbox_from_mesh(mesh_gt[:mano.vertex_num,:]) # xywh in original image space
                else:
                    bbox_gt = self.get_bbox_from_mesh(mesh_gt[mano.vertex_num:,:]) # xywh in original image space
                bbox_gt[2:] += bbox_gt[:2] # xywh -> xyxy
                
                bbox_out = bbox_out.reshape(2,2)
                bbox_out[:,0] = bbox_out[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                bbox_out[:,1] = bbox_out[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                bbox_out = np.concatenate((bbox_out, np.ones((2,1), dtype=np.float32)), 1)
                bbox_out = np.dot(bb2img_trans, bbox_out.transpose(1,0)).transpose(1,0)
                
                eval_result['bbox_iou'][2*n+idx] = get_iou(bbox_out, bbox_gt, 'xyxy')

            # mrrpe
            rel_trans_gt = np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:,:])[mano.sh_root_joint_idx] - np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num,:])[mano.sh_root_joint_idx]
            rel_trans_out = out['rel_trans'] * 1000 # meter to milimeter
            if annot['hand_type'] == 'interacting':
                eval_result['mrrpe'][n] = np.sqrt(np.sum((rel_trans_gt - rel_trans_out)**2))

            # root joint alignment
            for h in ('right', 'left'):
                if h == 'right':
                    vertex_mask = np.arange(0,mano.vertex_num)
                else:
                    vertex_mask = np.arange(mano.vertex_num,2*mano.vertex_num)
                mesh_gt[vertex_mask,:] = mesh_gt[vertex_mask,:] - np.dot(mano.sh_joint_regressor, mesh_gt[vertex_mask,:])[mano.sh_root_joint_idx,None,:]
                mesh_out[vertex_mask,:] = mesh_out[vertex_mask,:] - np.dot(mano.sh_joint_regressor, mesh_out[vertex_mask,:])[mano.sh_root_joint_idx,None,:]
            
            # mpvpe
            if annot['hand_type'] == 'right' and annot['right_mano_path'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
            elif annot['hand_type'] == 'left' and annot['left_mano_path'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()
            elif annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None:
                    eval_result['mpvpe_ih'][2*n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
                if annot['left_mano_path'] is not None:
                    eval_result['mpvpe_ih'][2*n+1] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()

            # mpvpe (right hand relative)
            if annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None and annot['left_mano_path'] is not None:
                    vertex_mask = np.arange(mano.vertex_num,2*mano.vertex_num)
                    mesh_gt[vertex_mask,:] = mesh_gt[vertex_mask,:] + rel_trans_gt
                    mesh_out[vertex_mask,:] = mesh_out[vertex_mask,:] + rel_trans_out
                    eval_result['rrve'][n] = np.sqrt(np.sum((mesh_gt - mesh_out)**2,1)).mean()

        return eval_result
    
    def print_eval_result(self, eval_result):
        tot_eval_result = {
                'mpvpe_sh': [],
                'mpvpe_ih': [],
                'rrve': [],
                'mrrpe': [],
                'bbox_iou': []
                }
        
        # mpvpe (average all samples)
        for mpvpe_sh in eval_result['mpvpe_sh']:
            if mpvpe_sh is not None:
                tot_eval_result['mpvpe_sh'].append(mpvpe_sh)
        for mpvpe_ih in eval_result['mpvpe_ih']:
            if mpvpe_ih is not None:
                tot_eval_result['mpvpe_ih'].append(mpvpe_ih)
        for mpvpe_ih in eval_result['rrve']:
            if mpvpe_ih is not None:
                tot_eval_result['rrve'].append(mpvpe_ih)
       
        # mrrpe (average all samples)
        for mrrpe in eval_result['mrrpe']:
            if mrrpe is not None:
                tot_eval_result['mrrpe'].append(mrrpe)
 
        # bbox IoU
        for iou in eval_result['bbox_iou']:
            if iou is not None:
                tot_eval_result['bbox_iou'].append(iou)
        
        # print evaluation results
        eval_result = tot_eval_result
        
        print('bbox IoU: %.2f' % (np.mean(eval_result['bbox_iou']) * 100))
        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
        print('RRVE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['rrve'])))

