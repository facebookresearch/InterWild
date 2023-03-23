import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import mano
from utils.preprocessing import load_img, process_bbox, augmentation, process_human_model_output
from utils.vis import vis_keypoints, vis_mesh, save_obj

class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        assert data_split == 'train', 'only training is supported for HIC dataset'
        self.data_path = osp.join('..', 'data', 'FreiHAND', 'data')
        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
        with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
            data = json.load(f)
            
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])
            
            body_bbox = np.array([0, 0, img['width'], img['height']], dtype=np.float32)
            body_bbox = process_bbox(body_bbox, img['width'], img['height'], extend_ratio=1.0)
            if body_bbox is None:
                continue

            cam_param, mano_param = data[db_idx]['cam_param'], {'right': data[db_idx]['mano_param'], 'left': None} # FreiHAND only contains right hand

            hand_bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if hand_bbox is None: continue

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'body_bbox': body_bbox,
                'rhand_bbox': hand_bbox,
                'lhand_bbox': None,
                'cam_param': cam_param,
                'mano_param': mano_param})
                
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
        img_path, img_shape, body_bbox = data['img_path'], data['img_shape'], data['body_bbox']
       
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # hand bbox transform
        lhand_bbox, lhand_bbox_valid = self.process_hand_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
        rhand_bbox, rhand_bbox_valid = self.process_hand_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1])/2.; rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1])/2.; 
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]; rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];

        # mano coordinates (right hand)
        mano_param = data['mano_param']
        mano_param['right']['hand_type'] = 'right'
        rmano_joint_img, rmano_joint_cam, rmano_joint_trunc, rmano_pose, rmano_shape, rmano_mesh_cam = process_human_model_output(mano_param['right'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
        rmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
        rmano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
        rmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)

        # dummy values
        lmano_joint_img = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
        lmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
        lmano_joint_trunc = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
        lmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
        lmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
        lmano_joint_valid = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
        lmano_pose_valid = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
        lmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)
        lmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)
        rel_trans = np.zeros((3), dtype=np.float32)
        rel_trans_valid = np.zeros((1), dtype=np.float32)

        if not do_flip:
            mano_joint_img = np.concatenate((rmano_joint_img, lmano_joint_img))
            mano_joint_cam = np.concatenate((rmano_joint_cam, lmano_joint_cam))
            mano_joint_trunc = np.concatenate((rmano_joint_trunc, lmano_joint_trunc))
            mano_pose = np.concatenate((rmano_pose, lmano_pose))
            mano_shape = np.concatenate((rmano_shape, lmano_shape))
            mano_joint_valid = np.concatenate((rmano_joint_valid, lmano_joint_valid))
            mano_pose_valid = np.concatenate((rmano_pose_valid, lmano_pose_valid))
            mano_shape_valid = np.concatenate((rmano_shape_valid, lmano_shape_valid))
            mano_mesh_cam = np.concatenate((rmano_mesh_cam, lmano_mesh_cam))
        else:
            mano_joint_img = np.concatenate((lmano_joint_img, rmano_joint_img))
            mano_joint_cam = np.concatenate((lmano_joint_cam, rmano_joint_cam))
            mano_joint_trunc = np.concatenate((lmano_joint_trunc, rmano_joint_trunc))
            mano_pose = np.concatenate((lmano_pose, rmano_pose))
            mano_shape = np.concatenate((lmano_shape, rmano_shape))
            mano_joint_valid = np.concatenate((lmano_joint_valid, rmano_joint_valid))
            mano_pose_valid = np.concatenate((lmano_pose_valid, rmano_pose_valid))
            mano_shape_valid = np.concatenate((lmano_shape_valid, rmano_shape_valid))
            mano_mesh_cam = np.concatenate((lmano_mesh_cam, rmano_mesh_cam))
        
        """
        # for debug
        _tmp = mano_joint_img.copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('freihand_' + str(idx) + '.jpg', _img)
        print('saved')
        """

        inputs = {'img': img}
        targets = {'joint_img': mano_joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': mano_joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_mesh_cam': mano_mesh_cam, 'rel_trans': rel_trans, 'mano_pose': mano_pose, 'mano_shape': mano_shape, 'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size}
        meta_info = {'bb2img_trans': bb2img_trans, 'joint_valid': mano_joint_valid, 'joint_trunc': mano_joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 'rel_trans_valid': rel_trans_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'lhand_bbox_valid': lhand_bbox_valid, 'rhand_bbox_valid': rhand_bbox_valid, 'is_3D': float(True)}
        return inputs, targets, meta_info


