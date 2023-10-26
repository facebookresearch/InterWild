# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D):
        loss = torch.abs(coord_out - coord_gt) * valid
        loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
        loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt, pose_valid):
        batch_size = pose_out.shape[0]

        pose_out = pose_out.view(batch_size,-1,3)
        pose_gt = pose_gt.view(batch_size,-1,3)

        #pose_out = matrix_to_axis_angle(axis_angle_to_matrix(pose_out))
        #pose_gt = matrix_to_axis_angle(axis_angle_to_matrix(pose_gt))

        #loss = torch.abs(pose_out - pose_gt) * pose_valid[:,:,None]

        pose_out = axis_angle_to_matrix(pose_out)
        pose_gt = axis_angle_to_matrix(pose_gt)

        loss = torch.abs(pose_out - pose_gt) * pose_valid[:,:,None,None]
        return loss


       
