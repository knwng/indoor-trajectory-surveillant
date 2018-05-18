#!/usr/bin/env python3

import os
import cv2
import sys
import math
import copy
import random
import numpy as np
import tensorflow as tf
from detector_nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

slim = tf.contrib.slim

class Detector(object):

    def __init__(self, ckpt='./checkpoints/ssd_300_vgg.ckpt', \
                       select_threshold=0.5, nms_threshold=0.45, \
                       num_classes=21, top_k=400):
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.ckpt = ckpt
        self.select_threshold = select_threshold
        self.nms_threshold = nms_threshold
        self.num_classes = num_classes
        self.top_k = top_k

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.graph = self.define_model()
        self.sess = self.load_model()

    def define_model(self):
        subgraph = tf.Graph()
        with subgraph.as_default():
            self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
            self.image_pre, self.labels_pre, \
                self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                                                            self.img_input, None, None, \
                                                            self.net_shape, self.data_format, \
                                                            resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(self.image_pre, 0)
            reuse = True if 'ssd_net' in locals() else None
            self.ssd_net = ssd_vgg_300.SSDNet()
            with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
                self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=reuse)
            self.ssd_anchors = self.ssd_net.anchors(self.net_shape)
            self.saver = tf.train.Saver()
        return subgraph

    def load_model(self):
        sess = tf.Session(config=self.config, graph=self.graph)
        # sess.run(tf.global_variables_initializer())
        self.saver.restore(sess, self.ckpt)
        return sess

    def detect(self, img, viz=False):
        img_h, img_w, img_c = img.shape

        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})

        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=self.select_threshold, img_shape=self.net_shape, num_classes=self.num_classes, decode=True)
    
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=self.top_k)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=self.nms_threshold)
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        rbboxes = [np.array([x[1]*img_w, x[3]*img_w, x[0]*img_h, x[2]*img_h]).astype(np.int) for idx, x in enumerate(rbboxes) if rclasses[idx] == 15]

        if viz:
            img_show = copy.deepcopy(img)
            for _bbox in rbboxes:
                cv2.rectangle(img_show, (_bbox[0], _bbox[2]), (_bbox[1], _bbox[3]), (0, 255, 0), 2)
            cv2.imshow('result', img_show)

        return rclasses, rscores, rbboxes

def video_test():
    # Test on some demo image and visualize output.
    path = '../videos/'  #input path
    output_path = '../outputs'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    video_name = 'hgh-test.avi'
    cap = cv2.VideoCapture(os.path.join(path, video_name))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 20.0, (960, 576))
    frame_idx = 1
    with open('bboxes.txt', 'w') as f: #store bboxes axis in txt file
        while(cap.isOpened()):
            ret, frame = cap.read()        
            if frame_idx % 4 == 0: #define frame sample rate
                if type(frame) == type(None): 
                    break
                height, width, _ = frame.shape
                rclasses, rscores, rbboxes =  process_image(frame)
                rbboxes = [x for idx, x in enumerate(rbboxes) if rclasses[idx] == 15]
                for bbox in rbboxes:
                    ymin = int(bbox[0] * height)
                    xmin = int(bbox[1] * width)
                    ymax = int(bbox[2] * height)
                    xmax = int(bbox[3] * width)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    out.write(frame)
                print('processing | frame {} | bbox {}'.format(frame_idx, rbboxes))
                # f.write('{} | {} {} {} {}\n'.format(frame_idx, xmin, xmax, ymin, ymax))
            frame_idx += 1
    out.release()
    cap.release()

def TEST():
    img = cv2.imread('./test.jpg')
    detector = Detector(ckpt='./checkpoints/ssd_300_vgg.ckpt')
    _classes, _scores, _bboxes = detector.detect(img, viz=True)
    print('class {} | score {} | bbox {}'.format(_classes, _scores, _bboxes))

if __name__ == '__main__':
    TEST()
    cv2.waitKey()
