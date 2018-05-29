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
                 detect_input_shape=512, top_k=5, reid_threshold=0.08,
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

    def single_detect(self, img, timestamp):
        rclasses, rscores, rbboxes = self.detector.detect(img, viz=False)
        if len(rbboxes) == 0:
            return [None]*5
        cropped_imgs = [cv2.resize(img[bbox[2]:bbox[3], bbox[0]:bbox[1], :], (128, 256)) for bbox in rbboxes]
        embs = self.reid.embed(cropped_imgs, augmentation=True, pre_crop_size=[144, 288])
        ids = []
        id_candidates = []
        if len(self.id_dict.keys()) == 0:
            for _emb in embs:
                self.id_dict[self.id_idx] = [_emb]
                ids.append(self.id_idx)
                id_candidates.append([self.id_idx, 1.0])
                self.id_idx += 1
        else:
            for _emb in embs:
                scores = []
                for key, value in self.id_dict.items():
                    for gal_emb in value:
                        scores.append([key, self.reid.score(_emb, gal_emb)])
                scores = np.array(scores)
                sorted_idx = [pick_idx for pick_idx in np.argsort(scores[:, 1])[::-1] if scores[pick_idx, 1] > self.reid_threshold]
                topk_idx = sorted_idx[:min(len(sorted_idx), self.top_k)]
                if len(topk_idx) == 0:
                    self.id_dict[self.id_idx] = [_emb]
                    ids.append(self.id_idx)
                    self.id_idx += 1
                else:
                    # print('top1 {}'.format(topk_idx[0]))
                    # print('current dict {}'.format(self.id_dict))
                    self.id_dict[scores[topk_idx[0], 0]].append(_emb)
                    ids.append(scores[topk_idx[0], 0])
                id_candidates.append(scores[topk_idx])
        return rbboxes, embs, ids, id_candidates, cropped_imgs

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
        if viz:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter(os.path.join(self.struct_storage_root, STORAGE_BASENAME+'_viz.mp4'), fourcc, 20.0, (960, 576))
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
            # height, width, _ = frame.shape
            # rclasses, rscores, rbboxes = self.detector.detect(frame, viz=False)
            rbboxes, embs, ids, id_candidates, cropped_imgs = self.single_detect(frame, frame_idx)
            if rbboxes is None:
                if viz:
                    output.write(frame)
                continue

            # if len(rbboxes) == 0:
            #     continue
            identity_idx = 1
            # cropped_imgs = [cv2.resize(frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :], (128, 256)) for bbox in rbboxes]
            # embs = self.reid.embed(cropped_imgs)
            raw_det_storage['frame-{}'.format(frame_idx)] = cropped_imgs
    
            for _bbox, _emb, _id, _idc in zip(rbboxes, embs, ids, id_candidates):
                cropped_img_url=''
                struct_info_storage['frames'].append(FrameInst(cropped_img_url=cropped_img_url,
                                                           timestamp=frame_idx,
                                                           bbox=_bbox,
                                                           identity=_id,
                                                           id_candidates=_idc,
                                                           embedding=_emb))
                if viz:
                    cv2.rectangle(frame, (_bbox[0], _bbox[2]), (_bbox[1], _bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, 'ID: {}'.format(int(_id)), (_bbox[0], _bbox[2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    output.write(frame)
                identity_idx += 1
            print('preparing | frame {} | id {}'.format(frame_idx, identity_idx-1))
        with open(os.path.join(self.struct_storage_root, STORAGE_BASENAME)+'.pkl', 'wb') as f:
            pickle.dump(struct_info_storage, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.raw_storage_root, STORAGE_BASENAME)+'.pkl', 'wb') as f:
            pickle.dump(raw_det_storage, f, pickle.HIGHEST_PROTOCOL)
        return

def TEST_JOINT_DET_REID():
    DETECT_INTERVAL = 4
    STRUCT_STORAGE_ROOT = './output/struct_storage'
    RAW_STORAGE_ROOT = './output/raw'

    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    video_dir = './videos/0509_1_A3_ELEVATOR_slice.mp4'
    # output_root_dir = os.path.basename(video_dir).split('.')[0]
    STORAGE_BASENAME = os.path.basename(video_dir).split('.')[0]
    if not os.path.exists(STRUCT_STORAGE_ROOT):
        os.makedirs(STRUCT_STORAGE_ROOT)
    if not os.path.exists(RAW_STORAGE_ROOT):
        os.makedirs(RAW_STORAGE_ROOT)
    # if not os.path.exists(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)):
    #     os.makedirs(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME))

    cap = cv2.VideoCapture(video_dir)
    frame_idx = 0
    struct_info_storage = {'video_fn':video_dir, 
                           'frames':[], 
                           'timestamp_offset':0.0,
                           'detect_interval':self.detect_interval,
                           'storage_dir':os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)}
    raw_det_storage = {}
    while cap.isOpened():
        frame_idx += 1
        if frame_idx % DETECT_INTERVAL != 0:
            continue
        ret, frame = cap.read()
        if frame is None:
            print('No frame stream')
            break
        height, width, _ = frame.shape
        rclasses, rscores, rbboxes = surve.detector.detect(frame, viz=False)
        if len(rbboxes) == 0:
            continue
        identity_idx = 1
        cropped_imgs = [cv2.resize(frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :], (128, 256)) for bbox in rbboxes]
        embs = surve.reid.embed(cropped_imgs)
        raw_det_storage['frame-{}'.format(frame_idx)] = cropped_imgs

        for _bbox, _emb in zip(rbboxes, embs):
            # cropped_img_url = os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME, 'frame-{}-idx-{}.jpg'.format(frame_idx, identity_idx))
            cropped_img_url=''
            struct_info_storage['frames'].append(FrameInst(cropped_img_url=cropped_img_url,
                                                           timestamp=frame_idx,
                                                           bbox=_bbox,
                                                           embedding=_emb))

            identity_idx += 1
        print('preparing | frame {} | id {}'.format(frame_idx, identity_idx-1))
    with open(os.path.join(STRUCT_STORAGE_ROOT, STORAGE_BASENAME)+'.pkl', 'wb') as f:
        pickle.dump(struct_info_storage, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(RAW_STORAGE_ROOT, STORAGE_BASENAME)+'.pkl', 'wb') as f:
        pickle.dump(raw_det_storage, f, pickle.HIGHEST_PROTOCOL)
    return

def TEST_REID():
    TOPK = 5
    GALLERY_ROOT = 'gallery_set_full'
    QUERY_ROOT = 'query_set'
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    gallery_fn_list = [os.path.join(GALLERY_ROOT, x) for x in os.listdir(GALLERY_ROOT) if x.endswith('.jpg')]
    query_fn_set = [os.path.join(QUERY_ROOT, x) for x in os.listdir(QUERY_ROOT) if x.endswith('.jpg')]
    print(gallery_fn_list)
    print(query_fn_set)
    print('load embedding into gallery')
    start = time.time()
    surve.load_gallery(gallery_fn_list)
    print('gallery loading time {} per image'.format((time.time() - start) / len(gallery_fn_list)))
    for query in query_fn_set:
        query_img = cv2.imread(query)
        surve.query(query_img, viz=True)
    cv2.waitKey()
    return

def TEST_PREPARE_ID():
    DETECT_INTERVAL = 4 
    # video_dir = './videos/0509_1_2_PASS_slice.mp4'
    video_dir = './videos/0509_1_A3_ELEVATOR_slice.mp4'
    output_root_dir = os.path.basename(video_dir)
    if not os.path.exists(output_root_dir):
        os.mkdir(output_root_dir)

    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    cap = cv2.VideoCapture(video_dir)
    frame_idx = 0
    while cap.isOpened():
        frame_idx += 1
        if frame_idx % DETECT_INTERVAL != 0:
            continue
        ret, frame = cap.read()
        if frame is None:
            print('No frame stream')
            break
        height, width, _ = frame.shape
        rclasses, rscores, rbboxes = surve.detector.detect(frame, viz=False)
        # print(rbboxes)
        if len(rbboxes) == 0:
            continue
        identity_idx = 1
        for bbox in rbboxes:
            id_img = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
            id_img = cv2.resize(id_img, (128, 256))
            cv2.imwrite(os.path.join(output_root_dir, 'frame-{}-id-{}.jpg'.format(frame_idx, identity_idx)), id_img)
            identity_idx += 1
        print('preparing | frame {} | id {}'.format(frame_idx, identity_idx-1))

def TEST_DET():
    img = cv2.imread("./test.jpg")
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
    surve.detector.detect(img, viz=True)
    cv2.waitKey()

if __name__ == '__main__':
    # TEST_JOINT_DET_REID()
    surve = Surveillant(detector_ckpt='cv_models/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000', detect_interval=1, reid_threshold=0.07)
    video_dir = './videos/0509_1_A3_ELEVATOR_slice.mp4'
    surve.video_structuralization(video_dir, viz=True)

