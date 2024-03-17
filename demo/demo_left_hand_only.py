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

cfg.set_args()
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
    img_path = osp.join(input_img_path, '%06d.png' % frame_idx)
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
    
    # get outputs
    hand_bbox = out['lhand_bbox'].cpu().numpy()[0].reshape(2,2) # xyxy
    joint_img = out['ljoint_img'].cpu().numpy()[0] # 2.5D pose
    mesh = out['lmano_mesh_cam'].cpu().numpy()[0] # root-relative mesh
    root_pose = out['lmano_root_pose'].cpu().numpy()[0] # MANO root pose
    hand_pose = out['lmano_hand_pose'].cpu().numpy()[0] # MANO hand pose
    shape = out['lmano_shape'].cpu().numpy()[0] # MANO shape parameter
    root_cam = out['lroot_cam'].cpu().numpy()[0] # 3D position of the root joint (wrist)
    
    # translation
    mesh = mesh + root_cam
    render_focal = out['render_lfocal'].clone()
    render_princpt = out['render_lprincpt'].clone()
        
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
    color = (102,255,102) # green
    vis_box = cv2.rectangle(original_img[:,:,::-1].copy(), (int(hand_bbox[0,0]), int(hand_bbox[0,1])), (int(hand_bbox[1,0]), int(hand_bbox[1,1])), color, 3)

    # 2D skeleton
    joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
    joint_img = np.dot(bb2img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
    color = (102,255,102) # green
    vis_skeleton = vis_keypoints_with_skeleton(original_img[:,:,::-1].copy(), joint_img, mano.sh_skeleton, color)

    # save mesh
    save_obj(mesh, mano.face['left'], osp.join(mesh_save_path, file_name + '_left.obj'))

    # save MANO parameters
    with open(osp.join(param_save_path, file_name + '_left.json'), 'w') as f:
        json.dump({'root_pose': root_pose.tolist(), 'hand_pose': hand_pose.tolist(), 'shape': shape.tolist(), 'root_trans': root_cam.tolist(), 'focal_ortho': render_focal.tolist(), 'princpt_ortho': render_princpt.tolist()}, f)

    # render
    with torch.no_grad():
        mesh = torch.from_numpy(mesh[None,:,:]).float().cuda()
        face = torch.from_numpy(mano.face['left'][None,:,:].astype(np.int32)).cuda()
        render_cam_params = {'focal': render_focal, 'princpt': render_princpt}
        rgb, depth = render_mesh_orthogonal(mesh, face, render_cam_params, (img_height,img_width), 'left')
        fg, is_fg = rgb[0].cpu().numpy(), (depth[0].cpu().numpy() > 0)
        bg = original_img[:,:,::-1].copy()
        render_out = fg * is_fg + bg * (1 - is_fg)

    # save box
    cv2.imwrite(osp.join(render_save_path, file_name + '_box.jpg'), vis_box)

    # save 2D skeleton
    cv2.imwrite(osp.join(render_save_path, file_name + '_skeleton.jpg'), vis_skeleton)

    # save render
    cv2.imwrite(osp.join(render_save_path, file_name + '_mesh.jpg'), render_out)

