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
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.vis import save_obj
from utils.human_models import mano

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    assert args.gpu_ids, print("Please set proper gpu ids")

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
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
input_img_path = './images'
box_save_path = './boxes'
mesh_save_path = './meshes'
param_save_path = './params'
os.makedirs(box_save_path, exist_ok=True)
os.makedirs(mesh_save_path, exist_ok=True)
os.makedirs(param_save_path, exist_ok=True)

img_path_list = glob(osp.join(input_img_path, '*.jpg')) + glob(osp.join(input_img_path, '*.png'))
for img_path in tqdm(img_path_list):
    file_name = img_path.split('/')[-1][:-4]

    original_img = load_img(img_path) 
    img_height, img_width = original_img.shape[:2]
    bbox = [0, 0, img_width, img_height]
    bbox = process_bbox(bbox, img_width, img_height)

    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    transform = transforms.ToTensor()
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    
    rel_trans = out['rel_trans'].cpu().numpy()[0] # 3D relative translation between two hands
    for h in ('right', 'left'):
        hand_bbox = out[h[0] + 'hand_bbox'].cpu().numpy()[0].reshape(2,2) # xyxy
        mesh = out[h[0] + 'mano_mesh_cam'].cpu().numpy()[0] # root-relative
        root_pose = out[h[0] + 'mano_root_pose'].cpu().numpy()[0] # MANO root pose
        hand_pose = out[h[0] + 'mano_hand_pose'].cpu().numpy()[0] # MANO hand pose
        shape = out[h[0] + 'mano_shape'].cpu().numpy()[0] # MANO shape parameter
        
        # bbox save
        hand_bbox[:,0] = hand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox[:,1] = hand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_bbox_xy1 = np.concatenate((hand_bbox, np.ones_like(hand_bbox[:,:1])),1)
        hand_bbox = np.dot(bb2img_trans, hand_bbox_xy1.transpose(1,0)).transpose(1,0)
        with open(osp.join(box_save_path, file_name + '_' + h + '.json'), 'w') as f:
            json.dump(hand_bbox.tolist(), f)

        # save mesh
        if h == 'right':
            save_obj(mesh, mano.face[h], osp.join(mesh_save_path, file_name + '_' + h + '.obj'))
        else:
            save_obj(mesh+rel_trans.reshape(1,3), mano.face[h], osp.join(mesh_save_path, file_name + '_' + h + '.obj'))

        # save MANO parameters
        with open(osp.join(param_save_path, file_name + '_' + h + '.json'), 'w') as f:
            if h == 'right':
                json.dump({'root_pose': root_pose.tolist(), 'hand_pose': hand_pose.tolist(), 'shape': shape.tolist(), 'root_trans': [0,0,0]}, f)
            else:
                 json.dump({'root_pose': root_pose.tolist(), 'hand_pose': hand_pose.tolist(), 'shape': shape.tolist(), 'root_trans': rel_trans.tolist()}, f)
   

