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
from utils.vis import save_obj, render_mesh, vis_keypoints_with_skeleton, vis_3d_skeleton
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
render_save_path = './renders'
mesh_save_path = './meshes'
param_save_path = './params'
vis_save_path = './vis'
os.makedirs(render_save_path, exist_ok=True)
os.makedirs(mesh_save_path, exist_ok=True)
os.makedirs(param_save_path, exist_ok=True)
os.makedirs(vis_save_path, exist_ok=True)

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
        root_pose = out[h[0] + 'mano_root_pose'].cpu().numpy()[0] # MANO root pose parameter
        hand_pose = out[h[0] + 'mano_hand_pose'].cpu().numpy()[0] # MANO hand pose parameter
        shape = out[h[0] + 'mano_shape'].cpu().numpy()[0] # MANO shape parameter
        root_cam = out[h[0] + 'mano_root_cam'].cpu().numpy()[0]
        joint_img = out['joint_img'].cpu().numpy()[0][mano.th_joint_type[h],:]
        
        """
        # render mesh
        hand_bbox[:,0] = hand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox[:,1] = hand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_bbox_xy1 = np.concatenate((hand_bbox, np.ones_like(hand_bbox[:,:1])),1)
        hand_bbox = np.dot(bb2img_trans, hand_bbox_xy1.transpose(1,0)).transpose(1,0)
        focal = [cfg.focal[0] / cfg.input_hand_shape[1] * (hand_bbox[1,0] - hand_bbox[0,0]),
                cfg.focal[1] / cfg.input_hand_shape[0] * (hand_bbox[1,1] - hand_bbox[0,1])]
        princpt = [cfg.princpt[0] / cfg.input_hand_shape[1] * (hand_bbox[1,0] - hand_bbox[0,0]) + hand_bbox[0,0],
                cfg.princpt[1] / cfg.input_hand_shape[0] * (hand_bbox[1,1] - hand_bbox[0,1]) + hand_bbox[0,1]]
        original_img = render_mesh(original_img, mesh + root_cam.reshape(1,3), mano.face[h], {'focal': focal, 'princpt': princpt})
        cv2.imwrite(osp.join(render_save_path, file_name + '.jpg'), original_img[:,:,::-1])
        """
        
        """
        # visualize 2D pose
        joint_img[:,0] = joint_img[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        joint_img[:,1] = joint_img[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
        joint_img = np.dot(bb2img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        vis_img = vis_keypoints_with_skeleton(original_img, joint_img, mano.sh_skeleton)
        cv2.imwrite(osp.join(vis_save_path, file_name + '_' + h + '.jpg'), vis_img[:,:,::-1])
        """
        
        # bbox save
        hand_bbox[:,0] = hand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        hand_bbox[:,1] = hand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        hand_bbox_xy1 = np.concatenate((hand_bbox, np.ones_like(hand_bbox[:,:1])),1)
        hand_bbox = np.dot(bb2img_trans, hand_bbox_xy1.transpose(1,0)).transpose(1,0)
        with open(file_name + '_' + h + '.json', 'w') as f:
            json.dump(hand_bbox.tolist(), f)

        # save mesh
        if h == 'right':
            save_obj(mesh, mano.face[h], osp.join(mesh_save_path, file_name + '_' + h + '.obj'))
        else:
            save_obj(mesh+rel_trans.reshape(1,3), mano.face[h], osp.join(mesh_save_path, file_name + '_' + h + '.obj'))

        # save MANO parameters
        #with open(osp.join(param_save_path, file_name + '_' + h + '.json'), 'w') as f:
        #    json.dump({'root_pose': root_pose.tolist(), 'hand_pose': hand_pose.tolist(), 'shape': shape.tolist()}, f)
    

