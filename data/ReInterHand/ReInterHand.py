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
from utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, transform_db_data, get_mano_data, transform_mano_data, get_iou, distort_projection_fisheye
from utils.transforms import world2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_keypoints, save_obj, vis_3d_skeleton
from glob import glob

class ReInterHand(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'ReInterHand', 'data')
        self.cam_mode = 'Mugsy_cameras' # Mugsy_cameras, Ego_cameras
        self.envmap_mode = 'envmap_per_frame' # envmap_per_frame, envmap_per_segment
        self.joint_set = {
                        'joint_num': 48,
                        'joints_name': ('R_Wrist', 'R_Thumb_0', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_0', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', 'R_Forearm_Stub', 'L_Wrist', 'L_Thumb_0', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_0', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', 'L_Forearm_Stub'),
                        'flip_pairs': [ (i,i+24) for i in range(24)]
                        }
        self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.joint_set['root_joint_idx'] = {'right': self.joint_set['joints_name'].index('R_Wrist'), 'left': self.joint_set['joints_name'].index('L_Wrist')}
        self.joint_set['invalid_joint_idxs'] = [self.joint_set['joints_name'].index(name) for name in ['R_Forearm_Stub', 'L_Forearm_Stub']]
        self.test_capture_ids = ['m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands', 'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands', 'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands']
        self.datalist = self.load_data()
        
    def load_data(self):
        datalist = []
        capture_id_list = [x.split('/')[-1] for x in glob(osp.join(self.data_path, '*'))]
        for capture_id in capture_id_list:
            if self.data_split == 'train':
                if capture_id in self.test_capture_ids:
                    continue
            else:
                if capture_id not in self.test_capture_ids:
                    continue

            # Mugsy_cameras + envmap_per_frame is rendered in 5 fps. cannot use 30 fps frame_list.txt
            if (self.cam_mode == 'Mugsy_cameras') and (self.envmap_mode == 'envmap_per_frame'):
                pass
            # Other settings are rendered in 30 fps. can use 30 fps frame_list.txt
            else:
                with open(osp.join(self.data_path, capture_id, 'frame_list.txt')) as f:
                    frame_idx_list = [int(x.split()[1]) for x in f.readlines()]
           
            # Mugsy_cameras
            if self.cam_mode == 'Mugsy_cameras':
                with open(osp.join(self.data_path, capture_id, 'Mugsy_cameras', 'cam_params.json')) as f:
                    cam_param = json.load(f)
                    for cam_name in cam_param.keys():
                        cam_param[cam_name] = {k: np.array(v, dtype=np.float32) for k,v in cam_param[cam_name].items()}
                for cam_name in cam_param.keys():
                    # Mugsy_cameras + envmap_per_frame is rendered in 5 fps. cannot use 30 fps frame_list.txt
                    if self.envmap_mode == 'envmap_per_frame':
                        frame_idx_list = [int(x.split('/')[-1][:-4]) for x in glob(osp.join(self.data_path, capture_id, 'Mugsy_cameras', self.envmap_mode, 'images', cam_name, '*'))]
                    for frame_idx in frame_idx_list:
                        img_path = osp.join(self.data_path, capture_id, 'Mugsy_cameras', self.envmap_mode, 'images', cam_name, '%06d.png' % frame_idx)
                        rhand_joint_path = osp.join(self.data_path, capture_id, 'orig_fits', 'right', 'Keypoints', 'keypoint-%06d.json' % frame_idx)
                        lhand_joint_path = osp.join(self.data_path, capture_id, 'orig_fits', 'left', 'Keypoints', 'keypoint-%06d.json' % frame_idx)
                        rhand_mano_param_path = osp.join(self.data_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_right.json')
                        lhand_mano_param_path = osp.join(self.data_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_left.json')
                        datalist.append({
                            'capture_id': capture_id,
                            'cam_name': cam_name,
                            'frame_idx': frame_idx,
                            'img_path': img_path,
                            'rhand_joint_path': rhand_joint_path,
                            'lhand_joint_path': lhand_joint_path,
                            'rhand_mano_param_path': rhand_mano_param_path, 
                            'lhand_mano_param_path': lhand_mano_param_path,
                            'cam_param': cam_param[cam_name]})
            # Ego_cameras
            elif self.cam_mode == 'Ego_cameras':
                with open(osp.join(self.data_path, capture_id, 'Ego_cameras', self.envmap_mode, 'truncation_ratio.json')) as f:
                    truncation_ratio = json.load(f)
                for frame_idx in frame_idx_list:
                    # do not use frames if too many joints are truncated
                    if truncation_ratio['%06d' % frame_idx] < 0.2:
                        continue
                    img_path = osp.join(self.data_path, capture_id, 'Ego_cameras', self.envmap_mode, 'images', '%06d.png' % frame_idx)
                    rhand_joint_path = osp.join(self.data_path, capture_id, 'orig_fits', 'right', 'Keypoints', 'keypoint-%06d.json' % frame_idx)
                    lhand_joint_path = osp.join(self.data_path, capture_id, 'orig_fits', 'left', 'Keypoints', 'keypoint-%06d.json' % frame_idx)
                    rhand_mano_param_path = osp.join(self.data_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_right.json')
                    lhand_mano_param_path = osp.join(self.data_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_left.json')
                    cam_param_path = osp.join(self.data_path, capture_id, 'Ego_cameras', self.envmap_mode, 'cam_params', '%06d.json' % frame_idx)
                    datalist.append({
                        'capture_id': capture_id,
                        'frame_idx': frame_idx,
                        'img_path': img_path, 
                        'rhand_joint_path': rhand_joint_path,
                        'lhand_joint_path': lhand_joint_path,
                        'rhand_mano_param_path': rhand_mano_param_path, 
                        'lhand_mano_param_path': lhand_mano_param_path,
                        'cam_param_path': cam_param_path})
        
        return datalist
                    
    def process_hand_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0,0,1,1], dtype=np.float32).reshape(2,2) # dummy value
            bbox_valid = float(False) # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2,2) 

            # flip augmentation
            if do_flip:
                bbox[:,0] = img_shape[1] - bbox[:,0] - 1
                bbox[0,0], bbox[1,0] = bbox[1,0].copy(), bbox[0,0].copy() # xmin <-> xmax swap
            
            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4,2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:,:1])),1) 
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            bbox[:,0] = bbox[:,0] / cfg.input_img_shape[1] * cfg.output_body_hm_shape[2]
            bbox[:,1] = bbox[:,1] / cfg.input_img_shape[0] * cfg.output_body_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:,0]); xmax = np.max(bbox[:,0]);
            ymin = np.min(bbox[:,1]); ymax = np.max(bbox[:,1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            
            bbox_valid = float(True)
            bbox = bbox.reshape(2,2)

        return bbox, bbox_valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        capture_id, frame_idx = data['capture_id'], data['frame_idx']
        img_path, rhand_joint_path, lhand_joint_path, rhand_mano_param_path, lhand_mano_param_path = data['img_path'], data['rhand_joint_path'], data['lhand_joint_path'], data['rhand_mano_param_path'], data['lhand_mano_param_path']
        if self.cam_mode == 'Mugsy_cameras':
            cam_param = data['cam_param']
        elif self.cam_mode == 'Ego_cameras':
            with open(data['cam_param_path']) as f:
                cam_param = json.load(f)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in cam_param.items()}
        cam_param['t'] /= 1000 # millimeter -> meter

        # img
        img = load_img(img_path)
        img_height, img_width, _ = img.shape
        img_shape = (img_height, img_width)
        body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
        body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # reih hand gt
        with open(rhand_joint_path) as f:
            rhand_joint_world = np.array(json.load(f), dtype=np.float32) / 1000 # millimeter to meter
        with open(lhand_joint_path) as f:
            lhand_joint_world = np.array(json.load(f), dtype=np.float32) / 1000 # millimeter to meter
        joint_world = np.concatenate((rhand_joint_world, lhand_joint_world))
        joint_valid = np.ones_like(joint_world[:,:1])
        joint_valid[self.joint_set['invalid_joint_idxs'],:] = 0 # mark invalid joints
        joint_cam = world2cam(joint_world, cam_param['R'], cam_param['t'])
        joint_cam[joint_cam[:,2]==0,2] = 1e-4 # prevent divide by zero
        if 'D' in cam_param:
            joint_img = distort_projection_fisheye(torch.from_numpy(joint_cam)[None], torch.from_numpy(cam_param['focal'])[None], torch.from_numpy(cam_param['princpt'])[None], torch.from_numpy(cam_param['D'])[None])
            joint_img = joint_img[0].numpy()[:,:2]
        else:
            joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])[:,:2]
        joint_trunc = joint_valid * ((joint_img[:,0] >= 0) * (joint_img[:,0] < img_width) * \
                    (joint_img[:,1] >= 0) * (joint_img[:,1] < img_height)).reshape(-1,1).astype(np.float32)
        if np.sum(joint_trunc[self.joint_set['joint_type']['right']]) > 0:
            rhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['right'],:], joint_trunc[self.joint_set['joint_type']['right'],0], extend_ratio=1.2)
            rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy
            rhand_bbox_orig = rhand_bbox.copy()
            rhand_exist = True
        else:
            rhand_bbox = None
            rhand_bbox_orig = np.array([0,0,1,1], dtype=np.float32)
            rhand_exist = False
        if np.sum(joint_trunc[self.joint_set['joint_type']['left']]) > 0:
            lhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['left'],:], joint_trunc[self.joint_set['joint_type']['left'],0], extend_ratio=1.2)
            lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy
            lhand_bbox_orig = lhand_bbox.copy()
            lhand_exist = True
        else:
            lhand_bbox = None
            lhand_bbox_orig = np.array([0,0,1,1], dtype=np.float32)
            lhand_exist = False

        # hand bbox transform
        lhand_bbox, lhand_bbox_valid = self.process_hand_bbox(lhand_bbox, do_flip, img_shape, img2bb_trans)
        rhand_bbox, rhand_bbox_valid = self.process_hand_bbox(rhand_bbox, do_flip, img_shape, img2bb_trans)
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1])/2.; rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1])/2.; 
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]; rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];

        # make all things root-relative and transform data
        rel_trans = joint_cam[self.joint_set['root_joint_idx']['left'],:] - joint_cam[self.joint_set['root_joint_idx']['right'],:]
        rel_trans_valid = joint_valid[self.joint_set['root_joint_idx']['left']] * joint_valid[self.joint_set['root_joint_idx']['right']]
        joint_cam[self.joint_set['joint_type']['right'],:] = joint_cam[self.joint_set['joint_type']['right'],:] - joint_cam[self.joint_set['root_joint_idx']['right'],None,:] # root-relative
        joint_cam[self.joint_set['joint_type']['left'],:] = joint_cam[self.joint_set['joint_type']['left'],:] - joint_cam[self.joint_set['root_joint_idx']['left'],None,:] # root-relative
        joint_img = np.concatenate((joint_img, joint_cam[:,2:]),1)
        joint_img, joint_cam, joint_valid, joint_trunc, rel_trans = transform_db_data(joint_img, joint_cam, joint_valid, rel_trans, do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], mano.th_joints_name)

        # mano coordinates (right hand)
        if rhand_exist:
            # mano coordinates (right hand)
            with open(rhand_mano_param_path) as f:
                mano_param = json.load(f)
            mano_param['hand_type'] = 'right'
            rmano_joint_img, rmano_joint_cam, rmano_mesh_cam, rmano_pose, rmano_shape = get_mano_data(mano_param, cam_param, do_flip, img_shape)
            rmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
            rmano_mesh_valid = np.ones((mano.vertex_num,1), dtype=np.float32)
            rmano_pose_valid = np.ones((mano.orig_joint_num), dtype=np.float32)
            rmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
        else:
            # dummy values
            rmano_joint_img = np.zeros((mano.sh_joint_num,2), dtype=np.float32)
            rmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            rmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)
            rmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
            rmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            rmano_joint_valid = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
            rmano_mesh_valid = np.zeros((mano.vertex_num,1), dtype=np.float32)
            rmano_pose_valid = np.zeros((mano.orig_joint_num), dtype=np.float32)
            rmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)

        # mano coordinates (left hand)
        if lhand_exist:
            # mano coordinates (left hand)
            with open(lhand_mano_param_path) as f:
                mano_param = json.load(f)
            mano_param['hand_type'] = 'left'
            lmano_joint_img, lmano_joint_cam, lmano_mesh_cam, lmano_pose, lmano_shape = get_mano_data(mano_param, cam_param, do_flip, img_shape)
            lmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
            lmano_mesh_valid = np.ones((mano.vertex_num,1), dtype=np.float32)
            lmano_pose_valid = np.ones((mano.orig_joint_num), dtype=np.float32)
            lmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
        else:
            # dummy values
            lmano_joint_img = np.zeros((mano.sh_joint_num,2), dtype=np.float32)
            lmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            lmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)
            lmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
            lmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            lmano_joint_valid = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
            lmano_mesh_valid = np.zeros((mano.vertex_num,1), dtype=np.float32)
            lmano_pose_valid = np.zeros((mano.orig_joint_num), dtype=np.float32)
            lmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)

        # change name when flip
        if do_flip:
            rmano_joint_img, lmano_joint_img = lmano_joint_img, rmano_joint_img
            rmano_joint_cam, lmano_joint_cam = lmano_joint_cam, rmano_joint_cam
            rmano_mesh_cam, lmano_mesh_cam = lmano_mesh_cam, rmano_mesh_cam
            rmano_pose, lmano_pose = lmano_pose, rmano_pose
            rmano_shape, lmano_shape = lmano_shape, rmano_shape
            rmano_joint_valid, lmano_joint_valid = lmano_joint_valid, rmano_joint_valid
            rmano_mesh_valid, lmano_mesh_valid = lmano_mesh_valid, rmano_mesh_valid
            rmano_pose_valid, lmano_pose_valid = lmano_pose_valid, rmano_pose_valid
            rmano_shape_valid, lmano_shape_valid = lmano_shape_valid, rmano_shape_valid

        # aggregate two-hand data
        mano_joint_img = np.concatenate((rmano_joint_img, lmano_joint_img))
        mano_joint_cam = np.concatenate((rmano_joint_cam, lmano_joint_cam))
        mano_mesh_cam = np.concatenate((rmano_mesh_cam, lmano_mesh_cam))
        mano_pose = np.concatenate((rmano_pose, lmano_pose))
        mano_shape = np.concatenate((rmano_shape, lmano_shape))
        mano_joint_valid = np.concatenate((rmano_joint_valid, lmano_joint_valid))
        mano_mesh_valid = np.concatenate((rmano_mesh_valid, lmano_mesh_valid))
        mano_pose_valid = np.concatenate((rmano_pose_valid, lmano_pose_valid))
        mano_shape_valid = np.concatenate((rmano_shape_valid, lmano_shape_valid))

        # make all things root-relative and transform data
        mano_joint_img = np.concatenate((mano_joint_img, mano_joint_cam[:,2:]),1) # 2.5D joint coordinates
        mano_joint_img[mano.th_joint_type['right'],2] -= mano_joint_cam[mano.th_root_joint_idx['right'],2]
        mano_joint_img[mano.th_joint_type['left'],2] -= mano_joint_cam[mano.th_root_joint_idx['left'],2]
        mano_mesh_cam[:mano.vertex_num,:] -= mano_joint_cam[mano.th_root_joint_idx['right'],None,:]
        mano_mesh_cam[mano.vertex_num:,:] -= mano_joint_cam[mano.th_root_joint_idx['left'],None,:]
        mano_joint_cam[mano.th_joint_type['right'],:] -= mano_joint_cam[mano.th_root_joint_idx['right'],None,:]
        mano_joint_cam[mano.th_joint_type['left'],:] -= mano_joint_cam[mano.th_root_joint_idx['left'],None,:]
        dummy_trans = np.zeros((3), dtype=np.float32)
        mano_joint_img, mano_joint_cam, mano_mesh_cam, mano_joint_trunc, _, mano_pose = transform_mano_data(mano_joint_img, mano_joint_cam, mano_mesh_cam, mano_joint_valid, dummy_trans, mano_pose, img2bb_trans, rot)
        
        """
        # for debug
        _tmp = joint_img[mano.th_joint_type['right'],:].copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('reih_' + str(idx) + '_gt_right.jpg', _img)
        # for debug
        _tmp = joint_img[mano.th_joint_type['left'],:].copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('reih_' + str(idx) + '_gt_left.jpg', _img)
        # for debug
        _tmp = mano_joint_img[mano.th_joint_type['right'],:].copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('reih_' + str(idx) + '_mano_right.jpg', _img)
        # for debug
        _tmp = mano_joint_img[mano.th_joint_type['left'],:].copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('reih_' + str(idx) + '_mano_left.jpg', _img)
        # for debug
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _tmp = lhand_bbox.copy().reshape(2,2)
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = cv2.rectangle(_img.copy(), (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
        cv2.imwrite('reih_' + str(idx) + '_lhand.jpg', _img)
        # for debug
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _tmp = rhand_bbox.copy().reshape(2,2)
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = cv2.rectangle(_img.copy(), (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
        cv2.imwrite('reih_' + str(idx) + '_rhand.jpg', _img)
        print('saved')
        """

        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': joint_cam, 'mano_mesh_cam': mano_mesh_cam, 'rel_trans': rel_trans, 'mano_pose': mano_pose, 'mano_shape': mano_shape, 'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size}
        meta_info = {'bb2img_trans': bb2img_trans, 'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_mesh_valid': mano_mesh_valid, 'rel_trans_valid': rel_trans_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'lhand_bbox_valid': lhand_bbox_valid, 'rhand_bbox_valid': rhand_bbox_valid, 'is_3D': float(True)}
        if self.data_split == 'test':
            targets['rhand_bbox'] = rhand_bbox_orig
            targets['lhand_bbox'] = lhand_bbox_orig
        return inputs, targets, meta_info
 
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'mpjpe': [[None for _ in range(mano.th_joint_num)] for _ in range(sample_num)],
                    'mpvpe': [None for _ in range(sample_num*2)],
                    'rrve': [None for _ in range(sample_num)],
                    'mrrpe': [None for _ in range(sample_num)],
                    'bbox_iou': [None for _ in range(sample_num*2)]
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            mesh_out = np.concatenate((out['rmano_mesh_cam'], out['lmano_mesh_cam'])) * 1000 # meter to milimeter
            mesh_gt = out['mano_mesh_cam_target'] * 1000 # meter to milimeter

            joint_out = np.concatenate((out['rmano_joint_cam'], out['lmano_joint_cam'])) * 1000 # meter to milimeter
            joint_gt = np.concatenate((np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num]),\
                                        np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:])))
            
            # visualize
            vis = False
            if vis:
                filename = str(annot['capture_id'] + '_' + annot['cam_name'] + '_' + str(annot['frame_idx']))
                img = out['img'].transpose(1,2,0)[:,:,::-1]*255
                
                """
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
                """
                cv2.imwrite(filename + '.jpg', img)

                save_obj(out['rmano_mesh_cam'], mano.face['right'], filename + '_right.obj')
                save_obj(out['lmano_mesh_cam'] + out['rel_trans'].reshape(1,3), mano.face['left'], filename + '_left.obj')

            # mrrpe
            rel_trans_gt = out['rel_trans_target'] * 1000 # meter to milimeter
            rel_trans_out = out['rel_trans'] * 1000 # meter to milimeter
            eval_result['mrrpe'][n] = np.sqrt(np.sum((rel_trans_gt - rel_trans_out)**2))

            # root joint alignment
            for h in ('right', 'left'):
                if h == 'right':
                    vertex_mask = np.arange(0,mano.vertex_num)
                else:
                    vertex_mask = np.arange(mano.vertex_num,2*mano.vertex_num)
                mesh_gt[vertex_mask,:] = mesh_gt[vertex_mask,:] - np.dot(mano.sh_joint_regressor, mesh_gt[vertex_mask,:])[mano.sh_root_joint_idx,None,:]
                mesh_out[vertex_mask,:] = mesh_out[vertex_mask,:] - np.dot(mano.sh_joint_regressor, mesh_out[vertex_mask,:])[mano.sh_root_joint_idx,None,:]
                joint_gt[mano.th_joint_type[h],:] = joint_gt[mano.th_joint_type[h],:] - joint_gt[mano.th_root_joint_idx[h],None,:]
                joint_out[mano.th_joint_type[h],:] = joint_out[mano.th_joint_type[h],:] - joint_out[mano.th_root_joint_idx[h],None,:]
            # mpjpe
            for j in range(mano.th_joint_num):
                eval_result['mpjpe'][n][j] = np.sqrt(np.sum((joint_out[j] - joint_gt[j])**2))
           
            # mpvpe 
            eval_result['mpvpe'][2*n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
            eval_result['mpvpe'][2*n+1] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()

            # mpvpe (right hand relative)
            vertex_mask = np.arange(mano.vertex_num,2*mano.vertex_num)
            mesh_gt[vertex_mask,:] = mesh_gt[vertex_mask,:] + rel_trans_gt
            mesh_out[vertex_mask,:] = mesh_out[vertex_mask,:] + rel_trans_out
            eval_result['rrve'][n] = np.sqrt(np.sum((mesh_gt - mesh_out)**2,1)).mean()

            # bbox IoU
            bb2img_trans = out['bb2img_trans']
            for idx, h in enumerate(('right', 'left')):
                bbox_out = out[h[0] + 'hand_bbox'] # xyxy in cfg.input_body_shape space
                bbox_gt = out[h[0] + 'hand_bbox_target'] # xyxy in original image space
                bbox_valid = out[h[0] + 'hand_bbox_valid']
                if not bbox_valid:
                    continue
                
                bbox_out = bbox_out.reshape(2,2)
                bbox_out[:,0] = bbox_out[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                bbox_out[:,1] = bbox_out[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                bbox_out = np.concatenate((bbox_out, np.ones((2,1), dtype=np.float32)), 1)
                bbox_out = np.dot(bb2img_trans, bbox_out.transpose(1,0)).transpose(1,0)
                
                eval_result['bbox_iou'][2*n+idx] = get_iou(bbox_out, bbox_gt, 'xyxy')

        return eval_result
    
    def print_eval_result(self, eval_result):
        tot_eval_result = {
                'mpjpe': [[] for _ in range(mano.th_joint_num)],
                'mpvpe': [],
                'rrve': [],
                'mrrpe': [],
                'bbox_iou': [],
                }
        
        # mpjpe (average all samples)
        for mpjpe in eval_result['mpjpe']:
            for j in range(mano.th_joint_num):
                if mpjpe[j] is not None:
                    tot_eval_result['mpjpe'][j].append(mpjpe[j])
        tot_eval_result['mpjpe'] = [np.mean(result) for result in tot_eval_result['mpjpe']]
        
        # mpvpe (average all samples)
        for mpvpe in eval_result['mpvpe']:
            if mpvpe is not None:
                tot_eval_result['mpvpe'].append(mpvpe)
        for mpvpe in eval_result['rrve']:
            if mpvpe is not None:
                tot_eval_result['rrve'].append(mpvpe)

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
        print('MPVPE: %.2f mm' % (np.mean(eval_result['mpvpe'])))
        print('RRVE: %.2f mm' % (np.mean(eval_result['rrve'])))
        print('MPJPE: %.2f mm' % (np.mean(eval_result['mpjpe'])))

