# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import BoxNet, HandRoI, PositionNet, RotationNet, TransNet
from nets.loss import CoordLoss, PoseLoss
from utils.mano import mano
from utils.transforms import restore_bbox
from config import cfg
import copy

class Model(nn.Module):
    def __init__(self, body_backbone, body_box_net, hand_roi_net, hand_position_net, hand_rotation_net, hand_trans_net):
        super(Model, self).__init__()
        self.body_backbone = body_backbone
        self.body_box_net = body_box_net

        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net
        self.hand_trans_net = hand_trans_net
        
        self.mano_layer_right = copy.deepcopy(mano.layer['right']).cuda()
        self.mano_layer_left = copy.deepcopy(mano.layer['left']).cuda()
        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()
 
        self.trainable_modules = [self.body_backbone, self.body_box_net, self.hand_roi_net, self.hand_position_net, self.hand_rotation_net, self.hand_trans_net]
 
    def get_coord(self, root_pose, hand_pose, shape, root_trans, hand_type):
        batch_size = root_pose.shape[0]
        zero_trans = torch.zeros((batch_size,3)).float().cuda()
        if hand_type == 'right':
            output = self.mano_layer_right(betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=zero_trans)
        else:
            output = self.mano_layer_left(betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=zero_trans)

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_cam = joint_cam[:,mano.sh_root_joint_idx,:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + root_trans[:,None,:]
        joint_cam = joint_cam - root_cam[:,None,:] + root_trans[:,None,:]

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:,:,1] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        y = y / cfg.input_hand_shape[0] * cfg.output_hand_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,mano.sh_root_joint_idx,:]
        joint_cam = joint_cam - root_cam[:,None,:]
        mesh_cam = mesh_cam - root_cam[:,None,:]
        return joint_proj, joint_cam, mesh_cam, root_cam

    def forward(self, inputs, targets, meta_info, mode):
        # body network
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape, mode='bilinear')
        body_feat = self.body_backbone(body_img)
        rhand_bbox_center, rhand_bbox_size, lhand_bbox_center, lhand_bbox_size, rhand_bbox_conf, lhand_bbox_conf = self.body_box_net(body_feat)
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach() # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach() # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        hand_feat, orig2hand_trans, hand2orig_trans = self.hand_roi_net(inputs['img'], rhand_bbox, lhand_bbox) # (2N, ...). right hand + flipped left hand
        
        # hand network
        joint_img = self.hand_position_net(hand_feat)
        mano_root_pose, mano_hand_pose, mano_shape, root_trans = self.hand_rotation_net(hand_feat, joint_img.detach())
        rhand_num, lhand_num = len(rhand_bbox), len(lhand_bbox)
        # restore flipped left hand joint coordinates
        rjoint_img = joint_img[:rhand_num,:,:]
        ljoint_img = joint_img[rhand_num:,:,:]
        ljoint_img_x = ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        ljoint_img_x = cfg.input_hand_shape[1] - 1 - ljoint_img_x
        ljoint_img_x = ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        ljoint_img = torch.cat((ljoint_img_x[:,:,None], ljoint_img[:,:,1:]),2)
        # restore flipped left root rotations
        rroot_pose = mano_root_pose[:rhand_num,:]
        lroot_pose = mano_root_pose[rhand_num:,:]
        lroot_pose = torch.cat((lroot_pose[:,0:1], -lroot_pose[:,1:3]),1)
        # restore flipped left hand joint rotations
        rhand_pose = mano_hand_pose[:rhand_num,:]
        lhand_pose = mano_hand_pose[rhand_num:,:].reshape(-1,mano.orig_joint_num-1,3) 
        lhand_pose = torch.cat((lhand_pose[:,:,0:1], -lhand_pose[:,:,1:3]),2).view(lhand_num,-1)
        # shape
        rshape = mano_shape[:rhand_num,:]
        lshape = mano_shape[rhand_num:,:]
        # restore flipped left root translation
        rroot_trans = root_trans[:rhand_num,:]
        lroot_trans = root_trans[rhand_num:,:]
        lroot_trans = torch.cat((-lroot_trans[:,0:1], lroot_trans[:,1:]),1)
        # affine transformation matrix
        rhand_orig2hand_trans = orig2hand_trans[:rhand_num]
        lhand_orig2hand_trans = orig2hand_trans[rhand_num:]
        rhand_hand2orig_trans = hand2orig_trans[:rhand_num]
        lhand_hand2orig_trans = hand2orig_trans[rhand_num:]

        # get outputs
        rjoint_proj, rjoint_cam, rmesh_cam, rroot_cam = self.get_coord(rroot_pose, rhand_pose, rshape, rroot_trans, 'right')
        ljoint_proj, ljoint_cam, lmesh_cam, lroot_cam = self.get_coord(lroot_pose, lhand_pose, lshape, lroot_trans, 'left')
       
        # relative translation
        rel_trans = self.hand_trans_net(rjoint_img.detach(), ljoint_img.detach(), rhand_hand2orig_trans.detach(), lhand_hand2orig_trans.detach())

        # combine outputs for the loss calculation (follow mano.th_joints_name)
        mano_pose = torch.cat((rroot_pose, rhand_pose, lroot_pose, lhand_pose),1)
        mano_shape = torch.cat((rshape, lshape),1)
        mesh_cam = torch.cat((rmesh_cam, lmesh_cam),1)
        joint_cam = torch.cat((rjoint_cam, ljoint_cam),1)
        joint_img = torch.cat((rjoint_img, ljoint_img),1)
        joint_proj = torch.cat((rjoint_proj, ljoint_proj),1)

        if mode == 'train':
            # loss functions
            loss = {}
            loss['rhand_bbox_center'] = torch.abs(rhand_bbox_center - targets['rhand_bbox_center']) * meta_info['rhand_bbox_valid'][:,None]
            loss['rhand_bbox_size'] = torch.abs(rhand_bbox_size - targets['rhand_bbox_size']) * meta_info['rhand_bbox_valid'][:,None]
            loss['lhand_bbox_center'] = torch.abs(lhand_bbox_center - targets['lhand_bbox_center']) * meta_info['lhand_bbox_valid'][:,None]
            loss['lhand_bbox_size'] = torch.abs(lhand_bbox_size - targets['lhand_bbox_size']) * meta_info['lhand_bbox_valid'][:,None]
            loss['rel_trans'] = torch.abs(rel_trans - targets['rel_trans']) * meta_info['rel_trans_valid']
            loss['mano_pose'] = self.pose_loss(mano_pose, targets['mano_pose'], meta_info['mano_pose_valid'])
            loss['mano_shape'] = torch.abs(mano_shape - targets['mano_shape']) * meta_info['mano_shape_valid']
            loss['joint_cam'] = torch.abs(joint_cam - targets['joint_cam']) * meta_info['joint_valid'] * meta_info['is_3D'][:,None,None] * 10
            loss['mano_mesh_cam'] = torch.abs(mesh_cam - targets['mano_mesh_cam']) * meta_info['mano_mesh_valid'] * 10
 
            # cfg.output_body_hm_shape -> cfg.output_hand_hm_shape
            for part_name, trans in (('right', rhand_orig2hand_trans), ('left', lhand_orig2hand_trans)):
                for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('mano_joint_img', 'mano_joint_trunc')):
                    x = targets[coord_name][:,mano.th_joint_type[part_name],0]
                    y = targets[coord_name][:,mano.th_joint_type[part_name],1]
                    z = targets[coord_name][:,mano.th_joint_type[part_name],2]
                    trunc = meta_info[trunc_name][:,mano.th_joint_type[part_name],0]

                    x = x / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
                    y = y / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
                    xy1 = torch.stack((x,y,torch.ones_like(x)),2)
                    xy = torch.bmm(trans, xy1.permute(0,2,1)).permute(0,2,1)

                    x, y = xy[:,:,0], xy[:,:,1]
                    x = x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
                    y = y / cfg.input_hand_shape[0] * cfg.output_hand_hm_shape[1]
                    z = z / cfg.output_body_hm_shape[0] * cfg.output_hand_hm_shape[0]
                    trunc *= ((x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (y < cfg.output_hand_hm_shape[1]))

                    coord = torch.stack((x,y,z),2)
                    trunc = trunc[:,:,None]
                    targets[coord_name] = torch.cat((targets[coord_name][:,:mano.th_joint_type[part_name][0],:], coord, targets[coord_name][:,mano.th_joint_type[part_name][-1]+1:,:]),1)
                    meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:,:mano.th_joint_type[part_name][0],:], trunc, meta_info[trunc_name][:,mano.th_joint_type[part_name][-1]+1:,:]),1)

            loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
            loss['mano_joint_img'] = torch.abs(joint_img - targets['mano_joint_img']) * meta_info['mano_joint_trunc']
            loss['joint_proj'] = torch.abs(joint_proj - targets['joint_img'][:,:,:2]) * meta_info['joint_trunc']
            return loss
        else:
            # cfg.output_hand_hm_shape -> cfg.input_img_shape
            for part_name, trans in (('right', rhand_hand2orig_trans), ('left', lhand_hand2orig_trans)):
                x = joint_proj[:,mano.th_joint_type[part_name],0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
                y = joint_proj[:,mano.th_joint_type[part_name],1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]

                xy1 = torch.stack((x, y, torch.ones_like(x)),2)
                xy = torch.bmm(trans, xy1.permute(0,2,1)).permute(0,2,1)
                joint_proj[:,mano.th_joint_type[part_name],0] = xy[:,:,0]
                joint_proj[:,mano.th_joint_type[part_name],1] = xy[:,:,1]

            # cfg.output_hand_hm_shape -> cfg.input_img_shape
            for part_name, trans in (('right', rhand_hand2orig_trans), ('left', lhand_hand2orig_trans)):
                x = joint_img[:,mano.th_joint_type[part_name],0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
                y = joint_img[:,mano.th_joint_type[part_name],1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]

                xy1 = torch.stack((x, y, torch.ones_like(x)),2)
                xy = torch.bmm(trans, xy1.permute(0,2,1)).permute(0,2,1)
                joint_img[:,mano.th_joint_type[part_name],0] = xy[:,:,0]
                joint_img[:,mano.th_joint_type[part_name],1] = xy[:,:,1]
                
            # warp focal lengths and princpts (right hand only)
            _joint_cam = joint_cam.clone()
            joint_idx = mano.th_joint_type['right']
            _joint_cam[:,joint_idx,:] += rroot_cam[:,None,:]
            scale_x = (torch.max(joint_proj[:,joint_idx,0],1)[0] - torch.min(joint_proj[:,joint_idx,0],1)[0]) / (torch.max(_joint_cam[:,joint_idx,0],1)[0] - torch.min(_joint_cam[:,joint_idx,0],1)[0])
            scale_y = (torch.max(joint_proj[:,joint_idx,1],1)[0] - torch.min(joint_proj[:,joint_idx,1],1)[0]) / (torch.max(_joint_cam[:,joint_idx,1],1)[0] - torch.min(_joint_cam[:,joint_idx,1],1)[0])
            trans_x = joint_proj[:,joint_idx,0].mean(1) - (_joint_cam[:,joint_idx,0] * scale_x[:,None]).mean(1)
            trans_y = joint_proj[:,joint_idx,1].mean(1) - (_joint_cam[:,joint_idx,1] * scale_y[:,None]).mean(1)
            render_rfocal = torch.stack((scale_x, scale_y),1)
            render_rprincpt = torch.stack((trans_x, trans_y),1)
 
            # warp focal lengths and princpts (left hand only)
            _joint_cam = joint_cam.clone()
            joint_idx = mano.th_joint_type['left']
            _joint_cam[:,joint_idx,:] += lroot_cam[:,None,:]
            scale_x = (torch.max(joint_proj[:,joint_idx,0],1)[0] - torch.min(joint_proj[:,joint_idx,0],1)[0]) / (torch.max(_joint_cam[:,joint_idx,0],1)[0] - torch.min(_joint_cam[:,joint_idx,0],1)[0])
            scale_y = (torch.max(joint_proj[:,joint_idx,1],1)[0] - torch.min(joint_proj[:,joint_idx,1],1)[0]) / (torch.max(_joint_cam[:,joint_idx,1],1)[0] - torch.min(_joint_cam[:,joint_idx,1],1)[0])
            trans_x = joint_proj[:,joint_idx,0].mean(1) - (_joint_cam[:,joint_idx,0] * scale_x[:,None]).mean(1)
            trans_y = joint_proj[:,joint_idx,1].mean(1) - (_joint_cam[:,joint_idx,1] * scale_y[:,None]).mean(1)
            render_lfocal = torch.stack((scale_x, scale_y),1)
            render_lprincpt = torch.stack((trans_x, trans_y),1)
              
            # warp focal lengths and princpts (two hand)i
            _joint_cam = joint_cam.clone()
            _joint_cam[:,mano.th_joint_type['right'],:] += rroot_cam[:,None,:]
            _joint_cam[:,mano.th_joint_type['left'],:] += (rroot_cam[:,None,:] + rel_trans[:,None,:])
            scale_x = (torch.max(joint_proj[:,:,0],1)[0] - torch.min(joint_proj[:,:,0],1)[0]) / (torch.max(_joint_cam[:,:,0],1)[0] - torch.min(_joint_cam[:,:,0],1)[0])
            scale_y = (torch.max(joint_proj[:,:,1],1)[0] - torch.min(joint_proj[:,:,1],1)[0]) / (torch.max(_joint_cam[:,:,1],1)[0] - torch.min(_joint_cam[:,:,1],1)[0])
            trans_x = joint_proj[:,:,0].mean(1) - (_joint_cam[:,:,0] * scale_x[:,None]).mean(1)
            trans_y = joint_proj[:,:,1].mean(1) - (_joint_cam[:,:,1] * scale_y[:,None]).mean(1)
            render_focal = torch.stack((scale_x, scale_y),1)
            render_princpt = torch.stack((trans_x, trans_y),1)

            # test output
            out = {}
            out['img'] = inputs['img']
            out['rel_trans'] = rel_trans
            out['rhand_bbox'] = restore_bbox(rhand_bbox_center, rhand_bbox_size, None, 1.0)
            out['lhand_bbox'] = restore_bbox(lhand_bbox_center, lhand_bbox_size, None, 1.0)
            out['rhand_bbox_conf'] = rhand_bbox_conf
            out['lhand_bbox_conf'] = lhand_bbox_conf
            out['rjoint_img'] = joint_img[:,mano.th_joint_type['right'],:]
            out['ljoint_img'] = joint_img[:,mano.th_joint_type['left'],:]
            out['rmano_mesh_cam'] = rmesh_cam
            out['lmano_mesh_cam'] = lmesh_cam
            out['rmano_joint_cam'] = rjoint_cam
            out['lmano_joint_cam'] = ljoint_cam
            out['rmano_root_pose'] = rroot_pose
            out['lmano_root_pose'] = lroot_pose
            out['rmano_hand_pose'] = rhand_pose
            out['lmano_hand_pose'] = lhand_pose
            out['rmano_shape'] = rshape
            out['lmano_shape'] = lshape
            out['rroot_cam'] = rroot_cam
            out['lroot_cam'] = lroot_cam
            out['render_rfocal'] = render_rfocal
            out['render_rprincpt'] = render_rprincpt
            out['render_lfocal'] = render_lfocal
            out['render_lprincpt'] = render_lprincpt
            out['render_focal'] = render_focal
            out['render_princpt'] = render_princpt
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
            if 'rel_trans' in targets:
                out['rel_trans_target'] = targets['rel_trans']
            if 'rhand_bbox' in targets:
                out['rhand_bbox_target'] = targets['rhand_bbox']
                out['lhand_bbox_target'] = targets['lhand_bbox']
            if 'rhand_bbox_valid' in meta_info:
                out['rhand_bbox_valid'] = meta_info['rhand_bbox_valid']
                out['lhand_bbox_valid'] = meta_info['lhand_bbox_valid']
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass

def get_model(mode):
    body_backbone = ResNetBackbone(cfg.body_resnet_type)
    body_box_net = BoxNet()

    hand_backbone = ResNetBackbone(cfg.hand_resnet_type)
    hand_roi_net = HandRoI(hand_backbone)
    hand_position_net = PositionNet()
    hand_rotation_net = RotationNet()

    hand_trans_backbone = ResNetBackbone(cfg.trans_resnet_type)
    hand_trans_net = TransNet(hand_trans_backbone)

    if mode == 'train':
        body_backbone.init_weights()
        body_box_net.apply(init_weights)

        hand_roi_net.apply(init_weights)
        hand_backbone.init_weights()
        hand_position_net.apply(init_weights)
        hand_rotation_net.apply(init_weights)

        hand_trans_net.apply(init_weights)
        hand_trans_backbone.init_weights()

    model = Model(body_backbone, body_box_net, hand_roi_net, hand_position_net, hand_rotation_net, hand_trans_net)
    return model
