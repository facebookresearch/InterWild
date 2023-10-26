# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 


import numpy as np
import torch
import os.path as osp
from config import cfg
from utils.transforms import transform_joint_to_other_db
import smplx

class MANO(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        self.layer = {'right': smplx.create(cfg.human_model_path, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False, **self.layer_arg), 'left': smplx.create(cfg.human_model_path, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False, **self.layer_arg)}
        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10
        
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(self.layer['left'].shapedirs[:,0,:] - self.layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:,0,:] *= -1

        # original MANO joint set
        self.orig_joint_num = 16
        self.orig_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.orig_root_joint_idx = self.orig_joints_name.index('Wrist')
        self.orig_flip_pairs = ()
        self.orig_joint_regressor = self.layer['right'].J_regressor.numpy() # same for the right and left hands

        # changed MANO joint set (single hands)
        self.sh_joint_num = 21 # manually added fingertips
        self.sh_joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.sh_skeleton = ( (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20) )
        self.sh_root_joint_idx = self.sh_joints_name.index('Wrist')
        self.sh_flip_pairs = ()
        # add fingertips to joint_regressor
        self.sh_joint_regressor = transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.sh_joints_name)
        self.sh_joint_regressor[self.sh_joints_name.index('Thumb_4')] = np.array([1 if i == 745 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Index_4')] = np.array([1 if i == 317 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Middle_4')] = np.array([1 if i == 445 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Ring_4')] = np.array([1 if i == 556 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.sh_joint_regressor[self.sh_joints_name.index('Pinky_4')] = np.array([1 if i == 673 else 0 for i in range(self.sh_joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)


        # changed MANO joint set (two hands)
        self.th_joint_num = 42 # manually added fingertips. two hands
        self.th_joints_name = ('R_Wrist', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', 'L_Wrist', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4')
        self.th_root_joint_idx = {'right': self.th_joints_name.index('R_Wrist'), 'left': self.th_joints_name.index('L_Wrist')}
        self.th_flip_pairs = [(i,i+21) for i in range(21)]
        self.th_joint_type = {'right': np.arange(0,self.th_joint_num//2), 'left': np.arange(self.th_joint_num//2,self.th_joint_num)}

mano = MANO()
