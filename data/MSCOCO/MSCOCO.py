# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.mano import mano
from utils.preprocessing import load_img, sanitize_bbox, process_bbox, augmentation, transform_db_data, transform_mano_data, get_mano_data, get_iou
from utils.transforms import transform_joint_to_other_db
from utils.vis import vis_keypoints, save_obj

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')

        # mscoco joint set
        self.joint_set = {
                        'joint_num': 42,
                        'joints_name': ('L_Wrist', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', 'R_Wrist', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4'),
                        'flip_pairs': [ (i,i+21) for i in range(21)],
                        }
        self.joint_set['joint_type'] = {'left': np.arange(0,self.joint_set['joint_num']//2), 'right': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.joint_set['root_joint_idx'] = {'left': self.joint_set['joints_name'].index('L_Wrist'), 'right': self.joint_set['joints_name'].index('R_Wrist')}
        self.datalist = self.load_data()
    
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
            with open(osp.join(self.annot_path, 'MSCOCO_train_MANO_NeuralAnnot.json')) as f:
                mano_params = json.load(f)
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            if self.data_split == 'train':
                imgname = osp.join('train2017', img['file_name'])
            else:
                imgname = osp.join('val2017', img['file_name'])
            img_path = osp.join(self.img_path, imgname)

            if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue
            if ann['lefthand_valid'] is False and ann['righthand_valid'] is False:
                continue
            
            # body bbox
            body_bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if body_bbox is None: continue
            
            # left hand bbox
            if ann['lefthand_valid'] is False:
                lhand_bbox = None
            else:
                lhand_bbox = np.array(ann['lefthand_box'], dtype=np.float32)
                lhand_bbox = sanitize_bbox(lhand_bbox, img['width'], img['height'])
            if lhand_bbox is not None:
                lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy

            # right hand bbox
            if ann['righthand_valid'] is False:
                rhand_bbox = None
            else:
                rhand_bbox = np.array(ann['righthand_box'], dtype=np.float32)
                rhand_bbox = sanitize_bbox(rhand_bbox, img['width'], img['height'])
            if rhand_bbox is not None:
                rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy

            joint_img = np.concatenate((
                                np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3),
                                np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)))
            joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
            joint_img = joint_img[:,:2]
            
            if self.data_split == 'train' and str(aid) in mano_params:
                mano_param = mano_params[str(aid)]
            else:
                mano_param = {'right': None, 'left': None}

            datalist.append({
                        'aid': aid,
                        'img_path': img_path, 
                        'img_shape': (img['height'],img['width']), 
                        'body_bbox': body_bbox,
                        'lhand_bbox': lhand_bbox, 
                        'rhand_bbox': rhand_bbox,
                        'joint_img': joint_img, 
                        'joint_valid': joint_valid, 
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
        
        # image load
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
 
        if self.data_split == 'train':
            # coco gt
            joint_img = np.concatenate((data['joint_img'], np.zeros_like(data['joint_img'][:,:1])),1)
            dummy_coord = np.zeros((self.joint_set['joint_num'],3), dtype=np.float32)
            dummy_trans = np.zeros((3), dtype=np.float32)
            rel_trans_valid = np.zeros((1), dtype=np.float32)
            joint_img, joint_cam, joint_valid, joint_trunc, rel_trans = transform_db_data(joint_img, dummy_coord, data['joint_valid'], dummy_trans, do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], mano.th_joints_name)

            # mano coordinates (right hand)
            mano_param = data['mano_param']
            if mano_param['right'] is not None:
                mano_param['right']['mano_param']['hand_type'] = 'right'
                rmano_joint_img, rmano_joint_cam, rmano_mesh_cam, rmano_pose, rmano_shape = get_mano_data(mano_param['right']['mano_param'], mano_param['right']['cam_param'], do_flip, img_shape)
                rmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
                rmano_mesh_valid = np.ones((mano.vertex_num,1), dtype=np.float32)
                rmano_pose_valid = np.ones((mano.orig_joint_num), dtype=np.float32)
                rmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
            else:
                # dummy values
                rmano_joint_img = np.zeros((mano.sh_joint_num,2), dtype=np.float32)
                rmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
                rmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)
                rmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
                rmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
                rmano_joint_valid = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
                rmano_mesh_valid = np.zeros((mano.vertex_num,1), dtype=np.float32)
                rmano_pose_valid = np.zeros((mano.orig_joint_num), dtype=np.float32)
                rmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)

            # mano coordinates (left hand)
            if mano_param['left'] is not None:
                mano_param['left']['mano_param']['hand_type'] = 'left'
                lmano_joint_img, lmano_joint_cam, lmano_mesh_cam, lmano_pose, lmano_shape = get_mano_data(mano_param['left']['mano_param'], mano_param['left']['cam_param'], do_flip, img_shape)
                lmano_joint_valid = np.ones((mano.sh_joint_num,1), dtype=np.float32)
                lmano_mesh_valid = np.ones((mano.vertex_num,1), dtype=np.float32)
                lmano_pose_valid = np.ones((mano.orig_joint_num), dtype=np.float32)
                lmano_shape_valid = np.ones((mano.shape_param_dim), dtype=np.float32)
            else:
                # dummy values
                lmano_joint_img = np.zeros((mano.sh_joint_num,2), dtype=np.float32)
                lmano_joint_cam = np.zeros((mano.sh_joint_num,3), dtype=np.float32)
                lmano_mesh_cam = np.zeros((mano.vertex_num,3), dtype=np.float32)
                lmano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
                lmano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
                lmano_joint_valid = np.zeros((mano.sh_joint_num,1), dtype=np.float32)
                lmano_mesh_valid = np.zeros((mano.vertex_num,1), dtype=np.float32)
                lmano_pose_valid = np.zeros((mano.orig_joint_num), dtype=np.float32)
                lmano_shape_valid = np.zeros((mano.shape_param_dim), dtype=np.float32)

            # change name when flip
            if do_flip:
                rmano_joint_img, lmano_joint_img = lmano_joint_img, rmano_joint_img
                rmano_joint_cam, lmano_joint_cam = lmano_joint_cam, rmano_joint_cam
                rmano_mesh_cam, lmano_mesh_cam = lmano_mesh_cam, rmano_mesh_cam
                rmano_pose, lmano_pose = lmano_pose, rmano_pose
                rmano_shape, lmano_shape = lmano_shape, rmano_shape
                rmano_joint_valid, lmano_joint_valid = lmano_joint_valid, rmano_joint_valid
                rmano_mesh_valid, lmano_mesh_valid = lmano_mesh_valid, rmano_mesh_valid
                rmano_pose_valid, lmano_pose_valid = lmano_pose_valid, rmano_pose_valid
                rmano_shape_valid, lmano_shape_valid = lmano_shape_valid, rmano_shape_valid

            # aggregate two-hand data
            mano_joint_img = np.concatenate((rmano_joint_img, lmano_joint_img))
            mano_joint_cam = np.concatenate((rmano_joint_cam, lmano_joint_cam))
            mano_mesh_cam = np.concatenate((rmano_mesh_cam, lmano_mesh_cam))
            mano_pose = np.concatenate((rmano_pose, lmano_pose))
            mano_shape = np.concatenate((rmano_shape, lmano_shape))
            mano_joint_valid = np.concatenate((rmano_joint_valid, lmano_joint_valid))
            mano_mesh_valid = np.concatenate((rmano_mesh_valid, lmano_mesh_valid))
            mano_pose_valid = np.concatenate((rmano_pose_valid, lmano_pose_valid))
            mano_shape_valid = np.concatenate((rmano_shape_valid, lmano_shape_valid))

            # make all depth root-relative and transform data
            mano_joint_img = np.concatenate((mano_joint_img, mano_joint_cam[:,2:]),1) # 2.5D joint coordinates
            mano_joint_img[mano.th_joint_type['right'],2] -= mano_joint_cam[mano.th_root_joint_idx['right'],2]
            mano_joint_img[mano.th_joint_type['left'],2] -= mano_joint_cam[mano.th_root_joint_idx['left'],2]
            mano_mesh_cam[:mano.vertex_num,:] -= mano_joint_cam[mano.th_root_joint_idx['right'],None,:]
            mano_mesh_cam[mano.vertex_num:,:] -= mano_joint_cam[mano.th_root_joint_idx['left'],None,:]
            mano_joint_cam[mano.th_joint_type['right'],:] -= mano_joint_cam[mano.th_root_joint_idx['right'],None,:]
            mano_joint_cam[mano.th_joint_type['left'],:] -= mano_joint_cam[mano.th_root_joint_idx['left'],None,:]
            dummy_trans = np.zeros((3), dtype=np.float32)
            mano_joint_img, mano_joint_cam, mano_mesh_cam, mano_joint_trunc, rel_trans, mano_pose = transform_mano_data(mano_joint_img, mano_joint_cam, mano_mesh_cam, mano_joint_valid, dummy_trans, mano_pose, img2bb_trans, rot)

            """
            # for debug
            _tmp = joint_img.copy()
            _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _img = vis_keypoints(_img.copy(), _tmp)
            cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
            # for debug
            _tmp = mano_joint_img.copy()
            _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _img = vis_keypoints(_img.copy(), _tmp)
            cv2.imwrite('coco_' + str(idx) + '_mano.jpg', _img)
            # for debug
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _tmp = lhand_bbox.copy().reshape(2,2)
            _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
            _img = cv2.rectangle(_img.copy(), (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
            cv2.imwrite('coco_' + str(idx) + '_lhand.jpg', _img)
            # for debug
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _tmp = rhand_bbox.copy().reshape(2,2)
            _tmp[:,0] = _tmp[:,0] / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
            _img = cv2.rectangle(_img.copy(), (int(_tmp[0,0]), int(_tmp[0,1])), (int(_tmp[1,0]), int(_tmp[1,1])), (255,0,0), 3)
            cv2.imwrite('coco_' + str(idx) + '_rhand.jpg', _img)
            print('saved')
            """

            inputs = {'img': img}
            targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': joint_cam, 'mano_mesh_cam': mano_mesh_cam, 'rel_trans': rel_trans, 'mano_pose': mano_pose, 'mano_shape': mano_shape, 'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size}
            meta_info = {'bb2img_trans': bb2img_trans, 'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_mesh_valid': mano_mesh_valid, 'rel_trans_valid': rel_trans_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'lhand_bbox_valid': lhand_bbox_valid, 'rhand_bbox_valid': rhand_bbox_valid, 'is_3D': float(False)}
            return inputs, targets, meta_info

        # test mode
        else:
            inputs = {'img': img}
            targets = {'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size, 'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'bbox_iou': [None for _ in range(sample_num*2)],
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # visualize
            vis = False
            if vis:
                filename = str(annot['aid'])
                img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()

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

                save_obj(out['rmano_mesh_cam'], mano.face['right'], filename + '_right.obj')
                save_obj(out['lmano_mesh_cam'] + out['rel_trans'].reshape(1,3), mano.face['left'], filename + '_left.obj')

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
                'bbox_iou': []
                }
        for iou in eval_result['bbox_iou']:
            if iou is not None:
                tot_eval_result['bbox_iou'].append(iou)

        eval_result = tot_eval_result
        print('bbox IoU: %.2f' % (np.mean(eval_result['bbox_iou']) * 100))



