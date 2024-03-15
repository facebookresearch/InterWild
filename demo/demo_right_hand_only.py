# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import json
import torch
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image, get_iou
from utils.vis import vis_keypoints_with_skeleton, save_obj, render_mesh_orthogonal
from utils.mano import mano

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    assert args.gpu_ids, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_6.pth'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare save paths
input_img_path = './images'
mesh_save_path = './meshes'
param_save_path = './params'
render_save_path = './renders'
os.makedirs(mesh_save_path, exist_ok=True)
os.makedirs(param_save_path, exist_ok=True)
os.makedirs(render_save_path, exist_ok=True)

# load paths of input images
img_path_list = glob(osp.join(input_img_path, '*.jpg')) + glob(osp.join(input_img_path, '*.png'))

# for each input image
for img_path in tqdm(img_path_list):
    file_name = img_path.split('/')[-1][:-4]
    
    # load image and make its aspect ratio follow cfg.input_img_shape
    original_img = load_img(img_path) 
    img_height, img_width = original_img.shape[:2]
    bbox = [0, 0, img_width, img_height]
    bbox = process_bbox(bbox, img_width, img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    transform = transforms.ToTensor()
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward to InterWild
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    
    # check IoU between boxes of two hands
    rhand_bbox = out['rhand_bbox'].cpu().numpy()[0]
    lhand_bbox = out['lhand_bbox'].cpu().numpy()[0]
    iou = get_iou(rhand_bbox, lhand_bbox, 'xyxy')
    is_th = False

    # for each right and left hand
    vis_box = original_img.copy()[:,:,::-1]
    vis_skeleton = original_img.copy()[:,:,::-1]
    prev_depth = None
    render_out = torch.flip(torch.from_numpy(original_img).float().cuda()[None,:,:,:], [3]) # batch_size, img_height, img_width, 3
    rroot_cam = out['rroot_cam'].cpu().numpy()[0] # 3D position of the right hand root joint (wrist)
    rel_trans = out['rel_trans'].cpu().numpy()[0] # 3D relative translation between two hands
    for h in ('right', 'left'):
      
        if h == 'left':
          continue
          
        # get outputs
        hand_bbox = out[h[0] + 'hand_bbox'].cpu().numpy()[0].reshape(2,2) # xyxy
        hand_bbox_conf = float(out[h[0] + 'hand_bbox_conf'].cpu().numpy()[0]) # bbox confidence
        joint_img = out[h[0] + 'joint_img'].cpu().numpy()[0] # 2.5D pose
        mesh = out[h[0] + 'mano_mesh_cam'].cpu().numpy()[0] # root-relative mesh
        root_pose = out[h[0] + 'mano_root_pose'].cpu().numpy()[0] # MANO root pose
        hand_pose = out[h[0] + 'mano_hand_pose'].cpu().numpy()[0] # MANO hand pose
        shape = out[h[0] + 'mano_shape'].cpu().numpy()[0] # MANO shape parameter
        root_cam = out[h[0] + 'root_cam'].cpu().numpy()[0] # 3D position of the root joint (wrist)
        
        # use rel_trans only when two-hand cases
        if is_th:
            if h == 'right':
                mesh = mesh + rroot_cam[None,:]
            else:
                mesh = mesh + rroot_cam[None,:] + rel_trans[None,:]
            render_focal = out['render_focal'].clone()
            render_princpt = out['render_princpt'].clone()
        else:
            mesh = mesh + root_cam
            render_focal = out['render_' + h[0] + 'focal'].clone()
            render_princpt = out['render_' + h[0] + 'princpt'].clone()
            
        # warp from cfg.input_img_shape to the orignal image space
        render_focal[:,0] = render_focal[:,0] / cfg.input_img_shape[1] * bbox[2]
        render_focal[:,1] = render_focal[:,1] / cfg.input_img_shape[0] * bbox[3]
        render_princpt[:,0] = render_princpt[:,0] / cfg.input_img_shape[1] * bbox[2] + bbox[0]
        render_princpt[:,1] = render_princpt[:,1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]

        # bbox save
        hand_bbox[:,0] = hand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox[:,1] = hand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_bbox_xy1 = np.concatenate((hand_bbox, np.ones_like(hand_bbox[:,:1])),1)
        hand_bbox = np.dot(bb2img_trans, hand_bbox_xy1.transpose(1,0)).transpose(1,0)
        if h == 'right':
            color = (255,0,255) # purple
        else:
            color = (102,255,102) # green
        vis_box = cv2.rectangle(vis_box.copy(), (int(hand_bbox[0,0]), int(hand_bbox[0,1])), (int(hand_bbox[1,0]), int(hand_bbox[1,1])), color, 3)

        # 2D skeleton
        joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
        joint_img = np.dot(bb2img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        if h == 'right':
            color = (255,0,255) # purple
        else:
            color = (102,255,102) # green
        vis_skeleton = vis_keypoints_with_skeleton(vis_skeleton, joint_img, mano.sh_skeleton, color)

        # save mesh
        save_obj(mesh, mano.face[h], osp.join(mesh_save_path, file_name + '_' + h + '.obj'))

        # save MANO parameters
        with open(osp.join(param_save_path, file_name + '_' + h + '.json'), 'w') as f:
            if h == 'right':
                json.dump({'root_pose': root_pose.tolist(), 'hand_pose': hand_pose.tolist(), 'shape': shape.tolist(), 'root_trans': [0,0,0]}, f)
            else:
                 json.dump({'root_pose': root_pose.tolist(), 'hand_pose': hand_pose.tolist(), 'shape': shape.tolist(), 'root_trans': rel_trans.tolist()}, f)

        # render
        with torch.no_grad():
            mesh = torch.from_numpy(mesh[None,:,:]).float().cuda()
            face = torch.from_numpy(mano.face[h][None,:,:].astype(np.int32)).cuda()
            render_cam_params = {'focal': render_focal, 'princpt': render_princpt}
            rgb, depth = render_mesh_orthogonal(mesh, face, render_cam_params, (img_height,img_width), h)
        valid_mask = (depth > 0)
        if prev_depth is None:
            render_mask = valid_mask.float()
            render_out = rgb * render_mask + render_out * (1 - render_mask)
            prev_depth = depth
        else:
            render_mask = (valid_mask * (((depth < prev_depth) + (prev_depth <= 0)) > 0)).float()
            render_out = rgb * render_mask + render_out * (1 - render_mask)
            prev_depth = depth * render_mask + prev_depth * (1 - render_mask)
 
    # save box
    cv2.imwrite(osp.join(render_save_path, file_name + '_box.jpg'), vis_box)

    # save 2D skeleton
    cv2.imwrite(osp.join(render_save_path, file_name + '_skeleton.jpg'), vis_skeleton)
   
    # save render
    cv2.imwrite(osp.join(render_save_path, file_name + '_mesh.jpg'), render_out[0].cpu().numpy())
