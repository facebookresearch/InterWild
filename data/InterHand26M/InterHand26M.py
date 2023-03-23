import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import math
import random
from glob import glob
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import mano
from utils.preprocessing import load_img, get_bbox, sanitize_bbox, process_bbox, augmentation, process_db_coord, process_human_model_output, get_iou
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_transform_3D, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_3d_skeleton

class InterHand26M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'InterHand26M', 'images')
        self.annot_path = osp.join('..', 'data', 'InterHand26M', 'annotations')

        # IH26M joint set
        self.joint_set = {
                        'joint_num': 42, 
                        'joints_name': ('R_Thumb_4', 'R_Thumb_3', 'R_Thumb_2', 'R_Thumb_1', 'R_Index_4', 'R_Index_3', 'R_Index_2', 'R_Index_1', 'R_Middle_4', 'R_Middle_3', 'R_Middle_2', 'R_Middle_1', 'R_Ring_4', 'R_Ring_3', 'R_Ring_2', 'R_Ring_1', 'R_Pinky_4', 'R_Pinky_3', 'R_Pinky_2', 'R_Pinky_1', 'R_Wrist', 'L_Thumb_4', 'L_Thumb_3', 'L_Thumb_2', 'L_Thumb_1', 'L_Index_4', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Middle_4', 'L_Middle_3', 'L_Middle_2', 'L_Middle_1', 'L_Ring_4', 'L_Ring_3', 'L_Ring_2', 'L_Ring_1', 'L_Pinky_4', 'L_Pinky_3', 'L_Pinky_2', 'L_Pinky_1', 'L_Wrist'),
                        'flip_pairs': [ (i,i+21) for i in range(21)]
                        }
        self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.joint_set['root_joint_idx'] = {'right': self.joint_set['joints_name'].index('R_Wrist'), 'left': self.joint_set['joints_name'].index('L_Wrist')}
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_data.json'))
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        
        with open('aid_human_annot_' + self.data_split + '.txt') as f:
            aid_list = f.readlines()
            aid_list = [int(x) for x in aid_list]

        datalist = []
        for aid in aid_list:
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            hand_type = ann['hand_type']

            # camera parameters
            t, R = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32).reshape(3), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32).reshape(2), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32).reshape(2)
            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}

            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_trunc = np.array(ann['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_trunc[self.joint_set['joint_type']['right']] *= joint_trunc[self.joint_set['root_joint_idx']['right']]
            joint_trunc[self.joint_set['joint_type']['left']] *= joint_trunc[self.joint_set['root_joint_idx']['left']]
            if np.sum(joint_trunc) == 0:
                continue

            joint_valid = np.array(joints[str(capture_id)][str(frame_idx)]['joint_valid'], dtype=np.float32).reshape(-1,1)
            joint_valid[self.joint_set['joint_type']['right']] *= joint_valid[self.joint_set['root_joint_idx']['right']]
            joint_valid[self.joint_set['joint_type']['left']] *= joint_valid[self.joint_set['root_joint_idx']['left']]
            if np.sum(joint_valid) == 0:
                continue

            # joint coordinates
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam = world2cam(joint_world, R, t)
            joint_cam[np.tile(joint_valid==0, (1,3))] = 1. # prevent zero division error
            joint_img = cam2pixel(joint_cam, focal, princpt)

            # body bbox
            body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
            if body_bbox is None:
                continue
            
            # left hand bbox
            if np.sum(joint_trunc[self.joint_set['joint_type']['left']]) == 0:
                lhand_bbox = None
            else:
                lhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['left'],:2], joint_trunc[self.joint_set['joint_type']['left'],0], extend_ratio=1.2)
                lhand_bbox = sanitize_bbox(lhand_bbox, img_width, img_height)
            if lhand_bbox is None:
                joint_valid[self.joint_set['joint_type']['left']] = 0
                joint_trunc[self.joint_set['joint_type']['left']] = 0
            else:
                lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy
            
            # right hand bbox
            if np.sum(joint_trunc[self.joint_set['joint_type']['right']]) == 0:
                rhand_bbox = None
            else:
                rhand_bbox = get_bbox(joint_img[self.joint_set['joint_type']['right'],:2], joint_trunc[self.joint_set['joint_type']['right'],0], extend_ratio=1.2)
                rhand_bbox = sanitize_bbox(rhand_bbox, img_width, img_height)
            if rhand_bbox is None:
                joint_valid[self.joint_set['joint_type']['right']] = 0
                joint_trunc[self.joint_set['joint_type']['right']] = 0
            else:
                rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy

            if lhand_bbox is None and rhand_bbox is None:
                continue

            # mano parameters
            try:
                mano_param = mano_params[str(capture_id)][str(frame_idx)].copy()
                if lhand_bbox is None:
                    mano_param['left'] = None
                if rhand_bbox is None:
                    mano_param['right'] = None
            except KeyError:
                mano_param = {'right': None, 'left': None}
 
            datalist.append({
                'aid': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'body_bbox': body_bbox,
                'lhand_bbox': lhand_bbox,
                'rhand_bbox': rhand_bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'joint_trunc': joint_trunc,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'hand_type': hand_type})
            
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
        data['cam_param']['t'] /= 1000 # milimeter to meter

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

        # ih26m hand gt
        joint_cam = data['joint_cam'] / 1000. # milimeter to meter
        joint_valid = data['joint_valid']
        rel_trans = joint_cam[self.joint_set['root_joint_idx']['left'],:] - joint_cam[self.joint_set['root_joint_idx']['right'],:]
        rel_trans_valid = joint_valid[self.joint_set['root_joint_idx']['left']] * joint_valid[self.joint_set['root_joint_idx']['right']]
        joint_cam[self.joint_set['joint_type']['right'],:] = joint_cam[self.joint_set['joint_type']['right'],:] - joint_cam[self.joint_set['root_joint_idx']['right'],None,:] # root-relative
        joint_cam[self.joint_set['joint_type']['left'],:] = joint_cam[self.joint_set['joint_type']['left'],:] - joint_cam[self.joint_set['root_joint_idx']['left'],None,:] # root-relative
        joint_img = data['joint_img']
        joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
        joint_img, joint_cam, joint_valid, joint_trunc, rel_trans = process_db_coord(joint_img, joint_cam, joint_valid, rel_trans, do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], mano.th_joints_name)

        # mano coordinates (right hand)
        mano_param = data['mano_param']
        if mano_param['right'] is not None:
            mano_param['right']['hand_type'] = 'right'
            rmano_joint_img, rmano_joint_cam, rmano_joint_trunc, rmano_pose, rmano_shape, rmano_mesh_cam = process_human_model_output(mano_param['right'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
            rmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
            rmano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
            rmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
        else:
            # dummy values
            rmano_joint_img = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            rmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
            rmano_joint_trunc = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
            rmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
            rmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            rmano_joint_valid = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
            rmano_pose_valid = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
            rmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)
            rmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)

        # mano coordinates (left hand)
        if mano_param['left'] is not None:
            mano_param['left']['hand_type'] = 'left'
            lmano_joint_img, lmano_joint_cam, lmano_joint_trunc, lmano_pose, lmano_shape, lmano_mesh_cam = process_human_model_output(mano_param['left'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
            lmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
            lmano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
            lmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
        else:
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
        _tmp = joint_img.copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('ih26m_' + str(idx) + '_' + data['hand_type'] + '.jpg', _img)
        # for debug
        _tmp = mano_joint_img.copy()
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_keypoints(_img.copy(), _tmp)
        cv2.imwrite('ih26m_' + str(idx) + '_' + data['hand_type'] + '_mano.jpg', _img)
        # for debug
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _tmp = lhand_bbox.copy().reshape(2,2)
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = cv2.rectangle(_img.copy(), (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
        cv2.imwrite('ih26m_' + str(idx) + data['hand_type'] + '_lhand.jpg', _img)
        # for debug
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _tmp = rhand_bbox.copy().reshape(2,2)
        _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
        _img = cv2.rectangle(_img.copy(), (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
        cv2.imwrite('ih26m_' + str(idx) + data['hand_type'] + '_rhand.jpg', _img)
        """

        inputs = {'img': img}
        targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_mesh_cam': mano_mesh_cam, 'rel_trans': rel_trans, 'mano_pose': mano_pose, 'mano_shape': mano_shape, 'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size}
        meta_info = {'bb2img_trans': bb2img_trans, 'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 'rel_trans_valid': rel_trans_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'lhand_bbox_valid': lhand_bbox_valid, 'rhand_bbox_valid': rhand_bbox_valid, 'is_3D': float(True)}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'mpjpe_sh': [[None for _ in range(self.joint_set['joint_num'])] for _ in range(sample_num)], 
                    'mpjpe_ih': [[None for _ in range(self.joint_set['joint_num'])] for _ in range(sample_num)],
                    'mpvpe_sh': [None for _ in range(sample_num)],
                    'mpvpe_ih': [None for _ in range(sample_num*2)],
                    'mrrpe': [None for _ in range(sample_num)],
                    'bbox_iou': [None for _ in range(sample_num*2)],
                    'mpvpe_aid_ih': [(None, None) for _ in range(sample_num*2)],
                    'mrrpe_aid_ih': [(None, None) for _ in range(sample_num)]
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            joint_gt = annot['joint_cam']
            joint_valid = annot['joint_trunc'].reshape(-1)

            out = outs[n]
            joint_out = transform_joint_to_other_db(np.concatenate((out['rmano_joint_cam'], out['lmano_joint_cam'])), mano.th_joints_name, self.joint_set['joints_name']) * 1000 # meter to milimeter
            mesh_out = np.concatenate((out['rmano_mesh_cam'], out['lmano_mesh_cam'])) * 1000 # meter to milimeter
            mesh_gt = out['mano_mesh_cam_target'] * 1000 # meter to milimeter
            
            # visualize
            vis = False
            if vis:
                filename = str(cur_sample_idx + n)
                img = out['img'].transpose(1,2,0)[:,:,::-1]*255

                joint_img = out['joint_img']
                ljoint_img = joint_img[mano.th_joint_type['left'],:]
                ljoint_img[:,0] = ljoint_img[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
                ljoint_img[:,1] = ljoint_img[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
                for j in range(len(ljoint_img)):
                    img = cv2.circle(img.copy(), (int(ljoint_img[j][0]), int(ljoint_img[j][1])), 3, (255,0,0), -1)
                for pair in mano.sh_skeleton:
                    i,j = pair
                    img = cv2.line(img.copy(), (int(ljoint_img[i][0]), int(ljoint_img[i][1])), (int(ljoint_img[j][0]), int(ljoint_img[j][1])), (255,0,0), 3)
                rjoint_img = joint_img[mano.th_joint_type['right'],:]
                rjoint_img[:,0] = rjoint_img[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
                rjoint_img[:,1] = rjoint_img[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
                for j in range(len(rjoint_img)):
                    img = cv2.circle(img.copy(), (int(rjoint_img[j][0]), int(rjoint_img[j][1])), 3, (0,0,255), -1)
                for pair in mano.sh_skeleton:
                    i,j = pair
                    img = cv2.line(img.copy(), (int(rjoint_img[i][0]), int(rjoint_img[i][1])), (int(rjoint_img[j][0]), int(rjoint_img[j][1])), (0,0,255), 3)

                lhand_bbox = out['lhand_bbox'].reshape(2,2).copy()
                lhand_bbox[:,0] = lhand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                lhand_bbox[:,1] = lhand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                lhand_bbox = lhand_bbox.reshape(4)
                img = cv2.rectangle(img.copy(), (int(lhand_bbox[0]), int(lhand_bbox[1])), (int(lhand_bbox[2]), int(lhand_bbox[3])), (255,0,0), 3)
                rhand_bbox = out['rhand_bbox'].reshape(2,2).copy()
                rhand_bbox[:,0] = rhand_bbox[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                rhand_bbox[:,1] = rhand_bbox[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                rhand_bbox = rhand_bbox.reshape(4)
                img = cv2.rectangle(img.copy(), (int(rhand_bbox[0]), int(rhand_bbox[1])), (int(rhand_bbox[2]), int(rhand_bbox[3])), (0,0,255), 3)
                cv2.imwrite(filename + '.jpg', img)

                #ljoint_img[:,2] = ljoint_img[:,2] / cfg.output_hand_hm_shape[2]
                #rjoint_img[:,2] = rjoint_img[:,2] / cfg.output_hand_hm_shape[2]

                #save_obj(out['rmano_mesh_cam'], mano.face['right'], filename + '_right.obj')
                #save_obj(out['lmano_mesh_cam'], mano.face['left'], filename + '_left.obj')
        
            # mrrpe
            rel_trans_gt = joint_gt[self.joint_set['root_joint_idx']['left']] - joint_gt[self.joint_set['root_joint_idx']['right']]
            rel_trans_out = out['rel_trans'] * 1000 # meter to milimeter
            if joint_valid[self.joint_set['root_joint_idx']['right']] * joint_valid[self.joint_set['root_joint_idx']['left']]:
                eval_result['mrrpe'][n] = np.sqrt(np.sum((rel_trans_gt - rel_trans_out)**2))
                eval_result['mrrpe_aid_ih'][n] = (float(eval_result['mrrpe'][n]), str(annot['aid']))

            # root joint alignment
            for h in ('right', 'left'):
                if h == 'right':
                    vertex_mask = np.arange(0,mano.vertex_num)
                else:
                    vertex_mask = np.arange(mano.vertex_num,2*mano.vertex_num)
                mesh_gt[vertex_mask,:] = mesh_gt[vertex_mask,:] - np.dot(mano.sh_joint_regressor, mesh_gt[vertex_mask,:])[mano.sh_root_joint_idx,None,:]
                mesh_out[vertex_mask,:] = mesh_out[vertex_mask,:] - np.dot(mano.sh_joint_regressor, mesh_out[vertex_mask,:])[mano.sh_root_joint_idx,None,:]
                joint_gt[self.joint_set['joint_type'][h],:] = joint_gt[self.joint_set['joint_type'][h],:] - joint_gt[self.joint_set['root_joint_idx'][h],None,:]
                joint_out[self.joint_set['joint_type'][h],:] = joint_out[self.joint_set['joint_type'][h],:] - joint_out[self.joint_set['root_joint_idx'][h],None,:]
            # mpjpe
            for j in range(self.joint_set['joint_num']):
                if joint_valid[j]:
                    if annot['hand_type'] == 'right' or annot['hand_type'] == 'left':
                        eval_result['mpjpe_sh'][n][j] = np.sqrt(np.sum((joint_out[j] - joint_gt[j])**2))
                    else:
                        eval_result['mpjpe_ih'][n][j] = np.sqrt(np.sum((joint_out[j] - joint_gt[j])**2))
           
            # mpvpe 
            if annot['hand_type'] == 'right' and annot['mano_param']['right'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
            elif annot['hand_type'] == 'left' and annot['mano_param']['left'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()
            elif annot['hand_type'] == 'interacting':
                if annot['mano_param']['right'] is not None:
                    eval_result['mpvpe_ih'][2*n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
                    eval_result['mpvpe_aid_ih'][2*n] = (float(eval_result['mpvpe_ih'][2*n]), str(annot['aid']))
                if annot['mano_param']['left'] is not None:
                    eval_result['mpvpe_ih'][2*n+1] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()
                    eval_result['mpvpe_aid_ih'][2*n+1] = (float(eval_result['mpvpe_ih'][2*n+1]), str(annot['aid']))

            # bbox IoU
            bb2img_trans = out['bb2img_trans']
            for idx, h in enumerate(('right', 'left')):
                bbox_out = out[h[0] + 'hand_bbox'] # xyxy in cfg.input_body_shape space
                bbox_gt = annot[h[0] + 'hand_bbox'] # xyxy in original image space
                if bbox_gt is None:
                    continue
                
                bbox_out = bbox_out.reshape(2,2)
                bbox_out[:,0] = bbox_out[:,0] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
                bbox_out[:,1] = bbox_out[:,1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
                bbox_out = np.concatenate((bbox_out, np.ones((2,1), dtype=np.float32)), 1)
                bbox_out = np.dot(bb2img_trans, bbox_out.transpose(1,0)).transpose(1,0)
                
                eval_result['bbox_iou'][2*n+idx] = get_iou(bbox_out, bbox_gt, 'xyxy')

        return eval_result
    
    def print_eval_result(self, eval_result):
        tot_eval_result = {
                'mpjpe_sh': [[] for _ in range(self.joint_set['joint_num'])], 
                'mpjpe_ih': [[] for _ in range(self.joint_set['joint_num'])],
                'mpvpe_sh': [],
                'mpvpe_ih': [],
                'mrrpe': [],
                'bbox_iou': [],
                }
        
        # mpjpe (average all samples)
        for mpjpe_sh in eval_result['mpjpe_sh']:
            for j in range(self.joint_set['joint_num']):
                if mpjpe_sh[j] is not None:
                    tot_eval_result['mpjpe_sh'][j].append(mpjpe_sh[j])
        tot_eval_result['mpjpe_sh'] = [np.mean(result) for result in tot_eval_result['mpjpe_sh']]
        for mpjpe_ih in eval_result['mpjpe_ih']:
            for j in range(self.joint_set['joint_num']):
                if mpjpe_ih[j] is not None:
                    tot_eval_result['mpjpe_ih'][j].append(mpjpe_ih[j])
        tot_eval_result['mpjpe_ih'] = [np.mean(result) for result in tot_eval_result['mpjpe_ih']]
        
        # mpvpe (average all samples)
        for mpvpe_sh in eval_result['mpvpe_sh']:
            if mpvpe_sh is not None:
                tot_eval_result['mpvpe_sh'].append(mpvpe_sh)
        for mpvpe_ih in eval_result['mpvpe_ih']:
            if mpvpe_ih is not None:
                tot_eval_result['mpvpe_ih'].append(mpvpe_ih)
        
        # mrrpe (average all samples)
        for mrrpe in eval_result['mrrpe']:
            if mrrpe is not None:
                tot_eval_result['mrrpe'].append(mrrpe)
        
        # bbox IoU
        for iou in eval_result['bbox_iou']:
            if iou is not None:
                tot_eval_result['bbox_iou'].append(iou)
        
        # print evaluation results
        eval_result = tot_eval_result
        
        print()
        print('bbox IoU: %.2f' % (np.mean(eval_result['bbox_iou']) * 100))
        print()

        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
        print()
        
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
        print()

        print('MPJPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'] + eval_result['mpjpe_ih'])))
        print('MPJPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_sh'])))
        print('MPJPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpjpe_ih'])))
        print()

