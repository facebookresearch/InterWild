import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import BoxNet, HandRoI, PositionNet, RotationNet, TransNet
from nets.loss import CoordLoss, ParamLoss
from utils.human_models import mano
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
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
        self.param_loss = ParamLoss()
 
        self.trainable_modules = [self.body_backbone, self.body_box_net, self.hand_roi_net, self.hand_position_net, self.hand_rotation_net, self.hand_trans_net]
    
    def forward_rotation_net(self, hand_feat, hand_coord):
        root_pose_6d, pose_param_6d, shape_param, cam_param = self.hand_rotation_net(hand_feat, hand_coord)

        # change 6d pose -> axis angles
        root_pose = rot6d_to_axis_angle(root_pose_6d).reshape(-1,3)
        pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
        cam_trans = self.get_camera_trans(cam_param)
        return root_pose, pose_param, shape_param, cam_trans

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_hand_shape[0]*cfg.input_hand_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def get_camera_trans_union(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_union_3d_size*cfg.camera_union_3d_size/(cfg.input_hand_shape[0]*cfg.input_hand_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def get_coord(self, root_pose, hand_pose, shape, cam_trans, cam_trans_from_union, hand_type):
        batch_size = root_pose.shape[0]
        if hand_type == 'right':
            output = self.mano_layer_right(betas=shape, hand_pose=hand_pose, global_orient=root_pose)
        else:
            output = self.mano_layer_left(betas=shape, hand_pose=hand_pose, global_orient=root_pose)

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.sh_joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,mano.sh_root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam

        # project 3D coordinates to 2D space (from single cropped hand image)
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2]) * cfg.focal[0] + cfg.princpt[0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2]) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        y = y / cfg.input_hand_shape[0] * cfg.output_hand_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        # project 3D coordinates to 2D space (from union hand image)
        x = (joint_cam[:,:,0] + cam_trans_from_union[:,None,0]) / (joint_cam[:,:,2] + cam_trans_from_union[:,None,2]) * cfg.focal[0] + cfg.princpt_union[0]
        y = (joint_cam[:,:,1] + cam_trans_from_union[:,None,1]) / (joint_cam[:,:,2] + cam_trans_from_union[:,None,2]) * cfg.focal[1] + cfg.princpt_union[1]
        x = x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        y = y / cfg.input_hand_shape[0] * cfg.output_hand_hm_shape[1]
        joint_proj_from_union = torch.stack((x,y),2)

        return joint_proj, joint_proj_from_union, joint_cam, mesh_cam, root_cam

    def forward(self, inputs, targets, meta_info, mode):
        # body network
        if cfg.use_gt_hand_bbox:
            lhand_bbox_center, lhand_bbox_size = targets['lhand_bbox_center'], targets['lhand_bbox_size']
            rhand_bbox_center, rhand_bbox_size = targets['rhand_bbox_center'], targets['rhand_bbox_size']
            lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 1.2).detach() # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
            rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 1.2).detach() # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        else:
            body_img = F.interpolate(inputs['img'], cfg.input_body_shape, mode='bilinear')
            body_feat = self.body_backbone(body_img)
            lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size = self.body_box_net(body_feat)
            lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach() # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
            rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach() # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        hand_feat = self.hand_roi_net(inputs['img'], lhand_bbox, rhand_bbox) # (2N, ...). flipped left hand + right hand
        
        # hand network
        joint_img = self.hand_position_net(hand_feat) # (2N, J_P, 3)
        mano_root_pose, mano_hand_pose, mano_shape, cam_trans = self.forward_rotation_net(hand_feat, joint_img.detach())
        lhand_num = len(lhand_bbox); rhand_num = len(rhand_bbox);
        # restore flipped left hand joint coordinates
        ljoint_img = joint_img[:lhand_num,:,:]
        ljoint_img = torch.cat((cfg.output_hand_hm_shape[2] - 1 - ljoint_img[:,:,0:1], ljoint_img[:,:,1:]),2)
        rjoint_img = joint_img[lhand_num:,:,:]
        # restore flipped left hand joint rotations
        lroot_pose = mano_root_pose[:lhand_num,:]
        lroot_pose = torch.cat((lroot_pose[:,0:1], -lroot_pose[:,1:3]),1)
        rroot_pose = mano_root_pose[lhand_num:,:]
        # restore flipped left hand joint rotations
        lhand_pose = mano_hand_pose[:lhand_num,:].reshape(-1,mano.orig_joint_num-1,3) 
        lhand_pose = torch.cat((lhand_pose[:,:,0:1], -lhand_pose[:,:,1:3]),2).view(lhand_num,-1)
        rhand_pose = mano_hand_pose[lhand_num:,:]
        # shape
        lshape = mano_shape[:lhand_num,:]
        rshape = mano_shape[lhand_num:,:]
        lshape = rshape # use the same shape for the left and right hands
        # restore flipped left camera translation
        lcam_trans = cam_trans[:lhand_num,:]
        lcam_trans = torch.cat((-lcam_trans[:,0:1], lcam_trans[:,1:]),1)
        rcam_trans = cam_trans[lhand_num:,:]

        # relative translation
        rcam_trans_from_union, rel_trans, hand_bbox_union = self.hand_trans_net(rjoint_img.detach(), ljoint_img.detach(), rhand_bbox.detach(), lhand_bbox.detach(), inputs['img'])
        rcam_trans_from_union = self.get_camera_trans_union(rcam_trans_from_union)
        lcam_trans_from_union = rcam_trans_from_union + rel_trans

        # get outputs
        ljoint_proj, ljoint_proj_from_union, ljoint_cam, lmesh_cam, lroot_cam = self.get_coord(lroot_pose, lhand_pose, lshape, lcam_trans, lcam_trans_from_union, 'left')
        rjoint_proj, rjoint_proj_from_union, rjoint_cam, rmesh_cam, rroot_cam = self.get_coord(rroot_pose, rhand_pose, rshape, rcam_trans, rcam_trans_from_union, 'right')
        
        # combine outputs for the loss calculation (follow mano.th_joints_name)
        mano_pose = torch.cat((rroot_pose, rhand_pose, lroot_pose, lhand_pose),1)
        mano_shape = torch.cat((rshape, lshape),1)
        joint_cam = torch.cat((rjoint_cam, ljoint_cam),1)
        joint_img = torch.cat((rjoint_img, ljoint_img),1)
        joint_proj = torch.cat((rjoint_proj, ljoint_proj),1)
        joint_proj_from_union = torch.cat((rjoint_proj_from_union, ljoint_proj_from_union),1)

        if mode == 'train':
            # loss functions
            loss = {}
            if not cfg.use_gt_hand_bbox:
                loss['lhand_bbox_center'] = self.coord_loss(lhand_bbox_center, targets['lhand_bbox_center'], meta_info['lhand_bbox_valid'][:,None]) 
                loss['lhand_bbox_size'] = self.coord_loss(lhand_bbox_size, targets['lhand_bbox_size'], meta_info['lhand_bbox_valid'][:,None])
                loss['rhand_bbox_center'] = self.coord_loss(rhand_bbox_center, targets['rhand_bbox_center'], meta_info['rhand_bbox_valid'][:,None]) 
                loss['rhand_bbox_size'] = self.coord_loss(rhand_bbox_size, targets['rhand_bbox_size'], meta_info['rhand_bbox_valid'][:,None])
            loss['rel_trans'] = self.coord_loss(rel_trans[:,None,:], targets['rel_trans'][:,None,:], meta_info['rel_trans_valid'][:,None,:])
            loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_param_valid'])
            loss['mano_shape'] = self.param_loss(mano_shape, targets['mano_shape'], meta_info['mano_shape_valid'])
            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None]) * 10
            loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'], meta_info['mano_joint_valid']) * 10

            # change hand target joint_img and joint_trunc according to hand bbox (cfg.output_body_hm_shape -> hand bbox space)
            targets['joint_img_from_union'] = targets['joint_img'][:,:,:2].clone()
            for part_name, bbox in (('left', lhand_bbox), ('right', rhand_bbox)):
                for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('mano_joint_img', 'mano_joint_trunc')):
                    x = targets[coord_name][:,mano.th_joint_type[part_name],0]
                    y = targets[coord_name][:,mano.th_joint_type[part_name],1]
                    z = targets[coord_name][:,mano.th_joint_type[part_name],2]
                    trunc = meta_info[trunc_name][:,mano.th_joint_type[part_name],0]
                    
                    x -= (bbox[:,None,0] / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2])
                    x *= (cfg.output_hand_hm_shape[2] / ((bbox[:,None,2] - bbox[:,None,0]) / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2]))
                    y -= (bbox[:,None,1] / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1])
                    y *= (cfg.output_hand_hm_shape[1] / ((bbox[:,None,3] - bbox[:,None,1]) / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1]))
                    z *= cfg.output_hand_hm_shape[0] / cfg.output_body_hm_shape[0]
                    trunc *= ((x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (y < cfg.output_hand_hm_shape[1]))
                    
                    coord = torch.stack((x,y,z),2)
                    trunc = trunc[:,:,None]
                    targets[coord_name] = torch.cat((targets[coord_name][:,:mano.th_joint_type[part_name][0],:], coord, targets[coord_name][:,mano.th_joint_type[part_name][-1]+1:,:]),1)
                    meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:,:mano.th_joint_type[part_name][0],:], trunc, meta_info[trunc_name][:,mano.th_joint_type[part_name][-1]+1:,:]),1)

            loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
            loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'], meta_info['mano_joint_trunc'])
            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_valid'])

            # change hand target joint_img and joint_trunc according to hand bbox (cfg.output_body_hm_shape -> hand bbox space)
            x = targets['joint_img_from_union'][:,:,0]
            y = targets['joint_img_from_union'][:,:,1]
            
            x -= (hand_bbox_union[:,None,0] / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2])
            x *= (cfg.output_hand_hm_shape[2] / ((hand_bbox_union[:,None,2] - hand_bbox_union[:,None,0]) / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2]))
            y -= (hand_bbox_union[:,None,1] / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1])
            y *= (cfg.output_hand_hm_shape[1] / ((hand_bbox_union[:,None,3] - hand_bbox_union[:,None,1]) / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1]))
            targets['joint_img_from_union'] = torch.stack((x,y),2) 
            loss['joint_proj_from_union'] = self.coord_loss(joint_proj_from_union, targets['joint_img_from_union'], meta_info['joint_valid'] * meta_info['is_th'][:,None,None])
            return loss
        else:
            # change hand output joint_img according to hand bbox
            for part_name, bbox in (('left', lhand_bbox), ('right', rhand_bbox)):
                joint_img[:,mano.th_joint_type[part_name],0] *= (((bbox[:,None,2] - bbox[:,None,0]) / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2]) / cfg.output_hand_hm_shape[2])
                joint_img[:,mano.th_joint_type[part_name],0] += (bbox[:,None,0] / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2])
                joint_img[:,mano.th_joint_type[part_name],1] *= (((bbox[:,None,3] - bbox[:,None,1]) / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1]) / cfg.output_hand_hm_shape[1])
                joint_img[:,mano.th_joint_type[part_name],1] += (bbox[:,None,1] / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1])

            # test output
            out = {}
            out['img'] = inputs['img']
            out['rel_trans'] = rel_trans
            out['lhand_bbox'] = restore_bbox(lhand_bbox_center, lhand_bbox_size, None, 1.0)
            out['rhand_bbox'] = restore_bbox(rhand_bbox_center, rhand_bbox_size, None, 1.0)
            out['joint_img'] = joint_img
            out['lmano_mesh_cam'] = lmesh_cam
            out['rmano_mesh_cam'] = rmesh_cam
            out['lmano_root_cam'] = lroot_cam
            out['rmano_root_cam'] = rroot_cam
            out['lmano_joint_cam'] = ljoint_cam
            out['rmano_joint_cam'] = rjoint_cam
            out['lmano_root_pose'] = lroot_pose
            out['rmano_root_pose'] = rroot_pose
            out['lmano_hand_pose'] = lhand_pose
            out['rmano_hand_pose'] = rhand_pose
            out['lmano_shape'] = lshape
            out['rmano_shape'] = rshape
            out['lcam_trans'] = lcam_trans
            out['rcam_trans'] = rcam_trans
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'mano_mesh_cam' in targets:
                out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
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
