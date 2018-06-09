#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os, sys
import cv2 
import json
import time
import pickle
import random

sys.path.insert(0, './reid')
from reid import ReIdentifier
sys.path.insert(0, './detector')
from detector import Detector
from structs import FrameInst

model_root = './cv_models'

class Surveillant(object):
    
    def __init__(self, detector_ckpt, reid_ckpt, detect_interval=8, 
                 detect_input_shape=512, top_k=5, reid_threshold=0.08, reid_register_threshold=0.15,
                 reid_model_name="resnet_v1_50", reid_head_name="fc1024",
                 struct_storage_root='./output/struct_storage', raw_storage_root='./output/raw'):
        self.detector = Detector(ckpt=os.path.join(os.getcwd(), detector_ckpt),
                                 net_shape=[detect_input_shape, detect_input_shape])
        self.detect_interval = detect_interval
        self.reid = ReIdentifier(model_name=reid_model_name, 
                                 head_name=reid_head_name,
                                 model_ckpt=os.path.join(os.getcwd(), reid_ckpt),
                                 input_size=[256, 128],
                                 surveillant_map=None)
        self.reid_threshold = reid_threshold
        self.reid_register_threshold = reid_register_threshold
        self.top_k = top_k
        self.struct_storage_root = struct_storage_root
        self.raw_storage_root = raw_storage_root
        self.id_dict = {}
        self.id_idx = int(0)
        if not os.path.exists(self.struct_storage_root):
            os.makedirs(self.struct_storage_root)
        if not os.path.exists(self.raw_storage_root):
            os.makedirs(self.raw_storage_root)
        # if not os.path.exists(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)):
        #     os.makedirs(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME))

    def reset_id(self):
        self.id_dict = {}
        self.id_idx = int(0)

    def single_detect(self, img, timestamp):
        rclasses, rscores, rbboxes = self.detector.detect(img, viz=False)
        if len(rbboxes) == 0:
            return [None]*6
        cropped_imgs = [cv2.resize(img[bbox[2]:bbox[3], bbox[0]:bbox[1], :], (128, 256)) for bbox in rbboxes]
        embs = self.reid.embed(cropped_imgs, augmentation=True, pre_crop_size=[144, 288])
        scores_candidates = []
        ids = []
        id_scores = []
        id_candidates = []
        if self.id_idx == 0:
            for _det_score, _emb in zip(rscores, embs):
                '''
                if _det_score < 0.8:
                    ids.append(-1)
                    id_candidates.append([-1, 0])
                    continue
                '''
                self.id_dict[self.id_idx] = [_emb]
                ids.append(self.id_idx)
                id_scores.append(1.0)
                id_candidates.append([self.id_idx, 1.0])
                self.id_idx += 1
        else:
            for _det_score, _emb in zip(rscores, embs):
                '''
                if _det_score < 0.8:
                    # ids.append(-1)
                    scores_candidates.append(np.array([[-1, 0] for _ in range(len(self.id_dict.keys()))]))
                    continue
                '''
                scores = []
                for key, value in self.id_dict.items():
                    score_same_id = []
                    for gal_emb in value:
                        score_same_id.append(self.reid.score(_emb, gal_emb))
                    scores.append([key, np.max(score_same_id)])
                scores = np.array(scores)
                scores_candidates.append(scores)

                '''
                sorted_idx = [pick_idx for pick_idx in np.argsort(scores[:, 1])[::-1] if scores[pick_idx, 1] > self.reid_threshold]
                topk_idx = sorted_idx[:min(len(sorted_idx), self.top_k)]
                if len(topk_idx) == 0:
                    self.id_dict[self.id_idx] = [_emb]
                    ids.append(self.id_idx)
                    self.id_idx += 1
                else:
                    self.id_dict[scores[topk_idx[0], 0]].append(_emb)
                    ids.append(scores[topk_idx[0], 0])
                id_candidates.append(scores[topk_idx])
                '''

            # scores_candidates is in shape of [query_emb_num, gallery_emb_num, 2] | each element is [id, score]
            # we should find the highest score of each feature in gallery, then find the top-k candidate for each query feature
            highest_idx = []
            for _score in zip(*scores_candidates):
                highest_idx.append(np.argmax([x[1] for x in _score]))
            # print('all candidates of score {}'.format(list(zip(*scores_candidates))))
            # print('highest idx {}'.format(highest_idx))
            highest_idx = np.array(highest_idx)
            for i, _emb in enumerate(embs):
                '''
                if scores_candidates[i][0][0] == -1:
                    ids.append(-1)
                    id_candidates.append([-1, 0])
                    continue
                '''
                pos_idx_mask = highest_idx == i
                # print('pos mask {}'.format(pos_idx_mask))
                c_scores = scores_candidates[i][pos_idx_mask]
                # print('candidate scores {}'.format(c_scores))
                sorted_idx = [pick_idx for pick_idx in np.argsort(c_scores[:, 1])[::-1] if c_scores[pick_idx, 1] > self.reid_threshold]
                topk_idx = sorted_idx[:min(len(sorted_idx), self.top_k)]
                if len(topk_idx) == 0:
                    # no matching; create new id
                    self.id_dict[self.id_idx] = [_emb]
                    ids.append(self.id_idx)
                    id_scores.append(1.0)
                    self.id_idx += 1
                else:
                    # to reduce noise, only when score >= reid_register_threshold, the embedding will be added into gallery
                    if c_scores[topk_idx[0], 0] >= self.reid_register_threshold:
                        self.id_dict[c_scores[topk_idx[0], 0]].append(_emb)
                    ids.append(c_scores[topk_idx[0], 0])
                    id_scores.append(c_scores[topk_idx[0], 1])
                id_candidates.append(c_scores[topk_idx])
        for _det_score, _id, _idc in zip(rscores, ids, id_candidates):
            print('detection scores {} | id {} | id candidates {}'.format(_det_score, _id, _idc))

        return rbboxes, embs, ids, id_scores, id_candidates, cropped_imgs

    def query(self, img, viz=False):
        emb = self.reid.embed(img, augmentation=True, pre_crop_size=[144, 288])
        scores = []
        for _identity in self.gallery:
            scores.append(self.reid.score(emb, _identity[1]))
        sorted_idx = [pick_idx for pick_idx in np.argsort(scores)[::-1] if scores[pick_idx] > self.reid_threshold]
        topk_idx = sorted_idx[:min(len(sorted_idx), self.top_k)]
        if viz:
            display_block = [img, np.zeros_like(img)]
            display_block = display_block + [cv2.imread(self.gallery[i][0]) for i in topk_idx]
            display_block = np.hstack(display_block)
            cv2.imshow(str(sum(emb[:5])), display_block)
        return
    
    def load_gallery(self, img_fn_list):
        img_list = [cv2.imread(_fn) for _fn in img_fn_list]
        embs = self.reid.embed(img_list)
        self.gallery = self.gallery + list(zip(img_fn_list, embs))
        return
    
    def video_structuralization(self, video_dir, viz=False):
        surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
        STORAGE_BASENAME = os.path.basename(video_dir).split('.')[0]
        cap = cv2.VideoCapture(video_dir)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

        if viz:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # output = cv2.VideoWriter(os.path.join(self.struct_storage_root, STORAGE_BASENAME+'_viz.mp4'), fourcc, 20.0, (960, 576))
            output = cv2.VideoWriter(os.path.join(self.struct_storage_root, STORAGE_BASENAME+'_viz.mp4'), fourcc, 20.0, (1280, 720))
        frame_idx = 0
        struct_info_storage = {'video_fn':video_dir, 
                               'frames':[], 
                               'timestamp_offset':0.0,
                               'storage_dir':os.path.join(self.raw_storage_root, STORAGE_BASENAME)}
        raw_det_storage = {}
        while cap.isOpened():
            frame_idx += 1
            if frame_idx % self.detect_interval != 0:
                if viz:
                    ret, frame = cap.read()
                    if frame is not None:
                        output.write(frame)
                continue
            ret, frame = cap.read()
            if frame is None:
                print('No frame stream')
                break
            rbboxes, embs, ids, id_scores, id_candidates, cropped_imgs = self.single_detect(frame, frame_idx)
            if rbboxes is None:
                print('preparing | frame {} | nothing detected'.format(frame_idx))
                if viz:
                    output.write(frame)
                continue

            identity_idx = 1
            raw_det_storage['frame-{}'.format(frame_idx)] = cropped_imgs
    
            for _bbox, _emb, _id, _ids, _idc in zip(rbboxes, embs, ids, id_scores, id_candidates):
                cropped_img_url=''
                struct_info_storage['frames'].append(FrameInst(cropped_img_url=cropped_img_url,
                                                           timestamp=frame_idx,
                                                           bbox=_bbox,
                                                           identity=_id,
                                                           id_candidates=_idc,
                                                           embedding=_emb))
                if viz:
                    cv2.rectangle(frame, (_bbox[0], _bbox[2]), (_bbox[1], _bbox[3]), (0, 255, 0), 2)
                    # cv2.putText(frame, 'ID: {}'.format(int(_id)), (_bbox[0], _bbox[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'ID {} | score {:.3f}'.format(int(_id), _ids), (_bbox[0], _bbox[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    output.write(frame)
                identity_idx += 1
        with open(os.path.join(self.struct_storage_root, STORAGE_BASENAME)+'.pkl', 'wb') as f:
            pickle.dump(struct_info_storage, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.raw_storage_root, STORAGE_BASENAME)+'.pkl', 'wb') as f:
            pickle.dump(raw_det_storage, f, pickle.HIGHEST_PROTOCOL)
        return

if __name__ == '__main__':
    # TEST_EXTRACT_FEATURE()
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000', detect_interval=1, reid_threshold=0.085, reid_register_threshold=0.12)
                        # reid_ckpt='cv_models/checkpoint-25000', detect_interval=1, reid_threshold=0.075, reid_register_threshold=0.10)
    video_root = './videos'
    '''
    videos = ['seq_{}.mp4'.format(i+1) for i in range(6)]
    for video_dir in videos:
        print('process {}'.format(video_dir))
        surve.reset_id()
        surve.video_structuralization(os.path.join(video_root, video_dir), viz=True)
    '''
    video_dir = 'seq_1.mp4'
    surve.video_structuralization(os.path.join(video_root, video_dir), viz=True)

