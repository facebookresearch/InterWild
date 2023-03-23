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
from utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, process_db_coord, process_human_model_output, get_iou, load_ply
from utils.transforms import rigid_transform_3D
from utils.vis import vis_keypoints, vis_mesh, save_obj

class HIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        assert data_split == 'test', 'only testing is supported for HIC dataset'
        self.data_path = osp.join('..', 'data', 'HIC', 'data')

        # HIC joint set
        self.joint_set = {
                        'joint_num': 28, 
                        'joints_name': ('R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Thumb_2', 'R_Thumb_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Thumb_2', 'L_Thumb_3'),
                        'flip_pairs': [ (i,i+14) for i in range(14)]
                        }
        self.joint_set['joint_type'] = {'right': np.arange(0,self.joint_set['joint_num']//2), 'left': np.arange(self.joint_set['joint_num']//2,self.joint_set['joint_num'])}
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.data_path, 'HIC.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.data_path, img['file_name'])
            hand_type = ann['hand_type']

            # bbox
            body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
            if body_bbox is None:
                continue

            # mano mesh
            if ann['right_mano_path'] is not None:
                right_mano_path = osp.join(self.data_path, ann['right_mano_path'])
            else:
                right_mano_path = None
            if ann['left_mano_path'] is not None:
                left_mano_path = osp.join(self.data_path, ann['left_mano_path'])
            else:
                left_mano_path = None

            datalist.append({
                'aid': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'body_bbox': body_bbox,
                'hand_type': hand_type,
                'right_mano_path': right_mano_path,
                'left_mano_path': left_mano_path})
            
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

        # mano coordinates
        right_mano_path = data['right_mano_path']
        if right_mano_path is not None:
            right_mesh = load_ply(right_mano_path)
        else:
            right_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        left_mano_path = data['left_mano_path']
        if left_mano_path is not None:
            left_mesh = load_ply(left_mano_path)
        else:
            left_mesh = np.zeros((mano.vertex_num, 3), dtype=np.float32)
        mano_mesh_cam = np.concatenate((right_mesh, left_mesh))
        
        inputs = {'img': img}
        targets = {'mano_mesh_cam': mano_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'mpvpe_sh': [None for _ in range(sample_num)],
                    'mpvpe_ih': [None for _ in range(sample_num*2)],
                    'mrrpe': [None for _ in range(sample_num)],
                    'mpvpe_aid_ih': [(None, None) for _ in range(sample_num*2)],
                    'mrrpe_aid_ih': [(None, None) for _ in range(sample_num)]
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            out = outs[n]
            mesh_out = np.concatenate((out['rmano_mesh_cam'], out['lmano_mesh_cam'])) * 1000 # meter to milimeter
            mesh_gt = out['mano_mesh_cam_target'] * 1000 # meter to milimeter
            
            # visualize
            vis = False
            if vis:
                filename = str(annot['aid'])
                """
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
                """

                save_obj(out['rmano_mesh_cam'], mano.face['right'], filename + '_right.obj')
                save_obj(out['lmano_mesh_cam'] + out['rel_trans'].reshape(1,3), mano.face['left'], filename + '_left.obj')
        
            # mrrpe
            rel_trans_gt = np.dot(mano.sh_joint_regressor, mesh_gt[mano.vertex_num:,:])[mano.sh_root_joint_idx] - np.dot(mano.sh_joint_regressor, mesh_gt[:mano.vertex_num,:])[mano.sh_root_joint_idx]
            rel_trans_out = out['rel_trans'] * 1000 # meter to milimeter
            if annot['hand_type'] == 'interacting':
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
            
            # mpvpe
            if annot['hand_type'] == 'right' and annot['right_mano_path'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
            elif annot['hand_type'] == 'left' and annot['left_mano_path'] is not None:
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()
            elif annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None:
                    eval_result['mpvpe_ih'][2*n] = np.sqrt(np.sum((mesh_gt[:mano.vertex_num,:] - mesh_out[:mano.vertex_num,:])**2,1)).mean()
                    eval_result['mpvpe_aid_ih'][2*n] = (float(eval_result['mpvpe_ih'][2*n]), str(annot['aid']))
                if annot['left_mano_path'] is not None:
                    eval_result['mpvpe_ih'][2*n+1] = np.sqrt(np.sum((mesh_gt[mano.vertex_num:,:] - mesh_out[mano.vertex_num:,:])**2,1)).mean()
                    eval_result['mpvpe_aid_ih'][2*n+1] = (float(eval_result['mpvpe_ih'][2*n+1]), str(annot['aid']))

        return eval_result
    
    def print_eval_result(self, eval_result):
        tot_eval_result = {
                'mpvpe_sh': [],
                'mpvpe_ih': [],
                'mrrpe': []
                }
        
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
 
        # print evaluation results
        eval_result = tot_eval_result
        
        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))

