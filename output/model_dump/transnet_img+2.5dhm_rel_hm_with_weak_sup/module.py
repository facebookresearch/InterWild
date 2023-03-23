import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_conv1d_layers, make_deconv_layers, make_linear_layers
from utils.human_models import mano
from utils.transforms import sample_joint_features, soft_argmax_1d, soft_argmax_2d, soft_argmax_3d
from config import cfg

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.depth_dim = cfg.output_hand_hm_shape[0]
        self.conv = make_conv_layers([2048, self.joint_num*self.depth_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, hand_feat):
        hand_hm = self.conv(hand_feat)
        _, _, height, width = hand_hm.shape
        hand_hm = hand_hm.view(-1,self.joint_num,self.depth_dim,height,width)
        hand_coord = soft_argmax_3d(hand_hm)
        return hand_coord

class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
        self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.pose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
        self.shape_out = make_linear_layers([2048, mano.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([2048, 3], relu_final=False)

    def forward(self, hand_feat, joint_img):
        batch_size = hand_feat.shape[0]
        
        # shape and camera parameters
        shape_param = self.shape_out(hand_feat.mean((2,3)))
        cam_param = self.cam_out(hand_feat.mean((2,3)))
        
        # pose
        hand_feat = self.conv(hand_feat)
        hand_feat = sample_joint_features(hand_feat, joint_img[:,:,:2]) # batch_size, joint_num, feat_dim
        hand_feat = torch.cat((hand_feat, joint_img),2).view(batch_size,-1)
        root_pose = self.root_pose_out(hand_feat)
        pose_param = self.pose_out(hand_feat)
        return root_pose, pose_param, shape_param, cam_param

class TransNet(nn.Module):
    def __init__(self, backbone):
        super(TransNet, self).__init__()
        self.joint_num = mano.sh_joint_num
        self.backbone = backbone
        self.conv = make_conv_layers([self.joint_num*2*cfg.input_hm_shape[0]+128,128], kernel=3, stride=1, padding=1)
        #self.rel_trans_fc = make_linear_layers([2*(512+2),3], relu_final=False)
        self.rel_trans_fc = make_linear_layers([2*(512+2),3*cfg.output_rel_hm_shape], relu_final=False)
        self.right_trans_fc = make_linear_layers([512,3], relu_final=False)

    def get_bbox(self, joint_img):
        x_img, y_img = joint_img[:,:,0], joint_img[:,:,1]
        xmin = torch.min(x_img,1)[0]; ymin = torch.min(y_img,1)[0]; xmax = torch.max(x_img,1)[0]; ymax = torch.max(y_img,1)[0];

        x_center = (xmin+xmax)/2.; width = xmax-xmin;
        xmin = x_center - 0.5 * width * 1.2
        xmax = x_center + 0.5 * width * 1.2
        
        y_center = (ymin+ymax)/2.; height = ymax-ymin;
        ymin = y_center - 0.5 * height * 1.2
        ymax = y_center + 0.5 * height * 1.2

        bbox = torch.stack((xmin, ymin, xmax, ymax),1).float().cuda()
        return bbox
    
    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.input_hm_shape[2])
        y = torch.arange(cfg.input_hm_shape[1])
        z = torch.arange(cfg.input_hm_shape[0])
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();
        
        x = joint_coord[:,:,0,None,None,None]; y = joint_coord[:,:,1,None,None,None]; z = joint_coord[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        heatmap = heatmap * 255
        return heatmap

    def set_aspect_ratio(self, bbox, aspect_ratio=1):
        # xyxy -> xywh
        bbox[:,2] = bbox[:,2] - bbox[:,0]
        bbox[:,3] = bbox[:,3] - bbox[:,1]
        
        w = bbox[:,2]
        h = bbox[:,3]
        c_x = bbox[:,0] + w/2.
        c_y = bbox[:,1] + h/2.

        # aspect ratio preserving bbox
        mask1 = w > (aspect_ratio * h)
        mask2 = w < (aspect_ratio * h)
        h[mask1] = w[mask1] / aspect_ratio
        w[mask2] = h[mask2] * aspect_ratio

        bbox[:,2] = w
        bbox[:,3] = h
        bbox[:,0] = c_x - bbox[:,2]/2.
        bbox[:,1] = c_y - bbox[:,3]/2.
        
        # xywh -> xyxy
        bbox[:,2] = bbox[:,2] + bbox[:,0]
        bbox[:,3] = bbox[:,3] + bbox[:,1]
        return bbox

    def forward(self, rjoint_img, ljoint_img, rhand_bbox, lhand_bbox, img):
        rjoint_img, ljoint_img, rhand_bbox, lhand_bbox = rjoint_img.clone(), ljoint_img.clone(), rhand_bbox.clone(), lhand_bbox.clone()

        # change hand output joint_img according to hand bbox (cfg.output_hand_hm_shape -> cfg.input_body_shape)
        for coord, bbox in ((ljoint_img, lhand_bbox), (rjoint_img, rhand_bbox)):
            coord[:,:,0] *= ((bbox[:,None,2] - bbox[:,None,0]) / cfg.output_hand_hm_shape[2])
            coord[:,:,0] += bbox[:,None,0]
            coord[:,:,1] *= ((bbox[:,None,3] - bbox[:,None,1]) / cfg.output_hand_hm_shape[1])
            coord[:,:,1] += bbox[:,None,1]
        
        # compute tight hand bboxes from joint_img
        rhand_bbox = self.get_bbox(rjoint_img) # xmin, ymin, xmax, ymax in cfg.input_body_shape
        lhand_bbox = self.get_bbox(ljoint_img) # xmin, ymin, xmax, ymax in cfg.input_body_shape

        # bbox union
        xmin = torch.minimum(lhand_bbox[:,0], rhand_bbox[:,0])
        ymin = torch.minimum(lhand_bbox[:,1], rhand_bbox[:,1])
        xmax = torch.maximum(lhand_bbox[:,2], rhand_bbox[:,2])
        ymax = torch.maximum(lhand_bbox[:,3], rhand_bbox[:,3])
        hand_bbox_union = torch.stack((xmin, ymin, xmax, ymax),1)
        hand_bbox_union = self.set_aspect_ratio(hand_bbox_union)
        
        # change hand target joint_img according to hand bbox (cfg.input_body_shape -> cfg.input_hm_shape)
        for coord in (rjoint_img, ljoint_img):
            coord[:,:,0] -= hand_bbox_union[:,None,0]
            coord[:,:,0] *= (cfg.input_hm_shape[2] / (hand_bbox_union[:,None,2] - hand_bbox_union[:,None,0]))
            coord[:,:,1] -= hand_bbox_union[:,None,1]
            coord[:,:,1] *= (cfg.input_hm_shape[1] / (hand_bbox_union[:,None,3] - hand_bbox_union[:,None,1]))
            coord[:,:,2] *= cfg.input_hm_shape[0] / cfg.output_hand_hm_shape[0]
 
        # hand heatmap
        rhand_hm = self.render_gaussian_heatmap(rjoint_img)
        rhand_hm = rhand_hm.view(-1,self.joint_num*cfg.input_hm_shape[0],cfg.input_hm_shape[1],cfg.input_hm_shape[2]) 
        lhand_hm = self.render_gaussian_heatmap(ljoint_img)
        lhand_hm = lhand_hm.view(-1,self.joint_num*cfg.input_hm_shape[0],cfg.input_hm_shape[1],cfg.input_hm_shape[2])
        hand_hm = torch.cat((rhand_hm, lhand_hm),1)

        hand_bbox_roi = torch.cat((torch.arange(hand_bbox_union.shape[0]).float().cuda()[:,None], hand_bbox_union),1) # batch_idx, xmin, ymin, xmax, ymax
        hand_bbox_roi[:,1] = hand_bbox_roi[:,1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox_roi[:,2] = hand_bbox_roi[:,2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_bbox_roi[:,3] = hand_bbox_roi[:,3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox_roi[:,4] = hand_bbox_roi[:,4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_img = torchvision.ops.roi_align(img, hand_bbox_roi, cfg.input_hand_shape, aligned=False) 

        # relative translation
        hand_feat = self.backbone(hand_img, stage='early')
        hand_feat = self.conv(torch.cat((hand_feat, hand_hm),1))
        hand_feat = self.backbone(hand_feat, stage='late')
        right_trans = self.right_trans_fc(hand_feat.mean((2,3)))

        wrist_img = torch.stack((rjoint_img[:,mano.sh_root_joint_idx,:], ljoint_img[:,mano.sh_root_joint_idx,:]),1)
        wrist_img = torch.stack((wrist_img[:,:,0] / 8, wrist_img[:,:,1] / 8),2)
        wrist_feat = sample_joint_features(hand_feat, wrist_img)
        wrist_feat = torch.cat((wrist_feat, wrist_img),2).view(-1,2*(512+2))
        rel_trans = self.rel_trans_fc(wrist_feat)
        rel_trans = soft_argmax_1d(rel_trans.view(-1,3,cfg.output_rel_hm_shape)) / cfg.output_rel_hm_shape * 2 - 1
        rel_trans = rel_trans * cfg.rel_trans_3d_size / 2
        return right_trans, rel_trans, hand_bbox_union

class BoxNet(nn.Module):
    def __init__(self):
        super(BoxNet, self).__init__()
        self.deconv = make_deconv_layers([2048,256,256,256])
        self.bbox_center = make_conv_layers([256,2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256,256,2], relu_final=False)
        self.rhand_size = make_linear_layers([256,256,2], relu_final=False)
       
    def forward(self, img_feat):
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center = bbox_center[:,0,:], bbox_center[:,1,:]
        
        # bbox size
        lhand_feat = sample_joint_features(img_feat, lhand_center.detach()[:,None,:])[:,0,:]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center.detach()[:,None,:])[:,0,:]
        rhand_size = self.rhand_size(rhand_feat)
        return lhand_center, lhand_size, rhand_center, rhand_size

class HandRoI(nn.Module):
    def __init__(self, backbone):
        super(HandRoI, self).__init__()
        self.backbone = backbone
                                           
    def forward(self, img, lhand_bbox, rhand_bbox):
        # left hand image crop and resize
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:,None], lhand_bbox),1) # batch_idx, xmin, ymin, xmax, ymax
        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:,1] = lhand_bbox_roi[:,1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:,2] = lhand_bbox_roi[:,2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_bbox_roi[:,3] = lhand_bbox_roi[:,3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:,4] = lhand_bbox_roi[:,4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_img = torchvision.ops.roi_align(img, lhand_bbox_roi, cfg.input_hand_shape, aligned=False) 
        lhand_img = torch.flip(lhand_img, [3]) # flip to the right hand

        # right hand image crop and resize
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:,None], rhand_bbox),1) # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:,1] = rhand_bbox_roi[:,1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:,2] = rhand_bbox_roi[:,2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_bbox_roi[:,3] = rhand_bbox_roi[:,3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:,4] = rhand_bbox_roi[:,4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_img = torchvision.ops.roi_align(img, rhand_bbox_roi, cfg.input_hand_shape, aligned=False) 

        hand_img = torch.cat((lhand_img, rhand_img))
        hand_feat = self.backbone(hand_img)
        return hand_feat
