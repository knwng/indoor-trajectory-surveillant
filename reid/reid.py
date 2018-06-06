#!/usr/bin/env python3

from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import loss
from loss import cdist
import os
import cv2
import time
import random

import h5py
import json
import numpy as np
import tensorflow as tf

from aggregators import AGGREGATORS
import common


def flip_augment(image):
    """ Returns both the original and the horizontal flip of an image. """
    flipped = cv2.flip(image, 1)
    return [image, flipped]


def five_crops(image, crop_size):
    """ Returns the central and four corner crops of `crop_size` from `image`. """
    image_size = tf.shape(image)[:2]
    crop_margin = tf.subtract(image_size, crop_size)
    assert_size = tf.assert_non_negative(
        crop_margin, message='Crop size must be smaller or equal to the image size.')
    with tf.control_dependencies([assert_size]):
        top_left = tf.floor_div(crop_margin, 2)
        bottom_right = tf.add(top_left, crop_size)
    center       = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    top_left     = image[:-crop_margin[0], :-crop_margin[1]]
    top_right    = image[:-crop_margin[0], crop_margin[1]:]
    bottom_left  = image[crop_margin[0]:, :-crop_margin[1]]
    bottom_right = image[crop_margin[0]:, crop_margin[1]:]
    return center, top_left, top_right, bottom_left, bottom_right


class ReIdentifier(object):

    def __init__(self, model_name, head_name, model_ckpt, input_size=(256, 128), surveillant_map=None):
        self.surveillant_map = surveillant_map
        # self.input_height = input_height
        # self.input_width = input_width
        self.input_size = list(input_size)
        self.graph = self.define_model(model_name, head_name)
        self.model_ckpt = model_ckpt
        self.sess = self.load_model()
        self.gallery = []

    def define_model(self, model_name, head_name):
        subgraph = tf.Graph()
        with subgraph.as_default():
            model = import_module('reid_nets.' + model_name)
            head = import_module('reid_heads.' + head_name)
            # self.image = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_WIDTH, INPUT_HEIGHT, 3], name='image')
            self.image = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_size, 3], name='image')
            
            self.endpoints, body_prefix = model.endpoints(self.image, is_training=False)
            with tf.name_scope('head'):
                self.endpoints = head.head(self.endpoints, 128, is_training=False)
            self.saver = tf.train.Saver()
        return subgraph

    def load_model(self):
        sess = tf.Session(graph=self.graph)
        self.saver.restore(sess, self.model_ckpt)
        return sess

    def embed(self, image, augmentation=False, pre_crop_size=None):
        if augmentation and pre_crop_size is None :
            print('Specify pre_crop_size and input_size for test time augmentation')
            raise ValueError
        if np.array(image).ndim == 3:
            image = image[np.newaxis, :]
        if augmentation:
            emb = []
            for img in image:
                # output: [288, 144, 3]
                img = cv2.resize(img, tuple(pre_crop_size))
                aug_img = self.testtime_augmentation(img, self.input_size)
                _emb = self.sess.run(self.endpoints['emb'], feed_dict={self.image:aug_img})
                emb.append(np.mean(_emb, axis=0))
        else:
            image = [cv2.resize(img, tuple(self.input_size[::-1])) for img in image]
            emb = self.sess.run(self.endpoints['emb'], feed_dict={self.image:image})
        return emb

    def testtime_augmentation(self, image, crop_size):
        # horizontal flip
        flipped = cv2.flip(image, 1)
        image_size = image.shape[:2]
        crop_margin = np.subtract(image_size, crop_size)
        assert(crop_margin[0] >= 0 and crop_margin[1] >= 0)
        p_top_left = np.floor_divide(crop_margin, 2).astype(np.int)
        p_bottom_right = p_top_left + crop_size
        
        center       = image[p_top_left[0]:p_bottom_right[0], p_top_left[1]:p_bottom_right[1], :]
        top_left     = image[:-crop_margin[0], :-crop_margin[1], :]
        top_right    = image[:-crop_margin[0], crop_margin[1]:, :]
        bottom_left  = image[crop_margin[0]:, :-crop_margin[1], :]
        bottom_right = image[crop_margin[0]:, crop_margin[1]:, :]

        f_center       = flipped[p_top_left[0]:p_bottom_right[0], p_top_left[1]:p_bottom_right[1], :]
        f_top_left     = flipped[:-crop_margin[0], :-crop_margin[1], :]
        f_top_right    = flipped[:-crop_margin[0], crop_margin[1]:, :]
        f_bottom_left  = flipped[crop_margin[0]:, :-crop_margin[1], :]
        f_bottom_right = flipped[crop_margin[0]:, crop_margin[1]:, :]

        return np.stack([cv2.resize(image, tuple(crop_size[::-1])), cv2.resize(flipped, tuple(crop_size[::-1])), 
                center, top_left, top_right, bottom_left, bottom_right,
                f_center, f_top_left, f_top_right, f_bottom_left, f_bottom_right], axis=0)

    def score(self, ftr1, ftr2, single_batch=True, metric='euclidean'):
        if single_batch:
            return 1. / (1. + np.sqrt(np.sum((ftr1 - ftr2) ** 2)))
        else:
            raise NotImplementedError


def prepare_image(img_dir, height, width):
    img = cv2.imread(img_dir)
    # img = cv2.resize(img, (height, width))
    return img

def show_stats(name, x):
    print('stats of {} | max {} | mean {} | min {}'.format(name, np.max(x), np.mean(x), np.min(x)))
    return

def im_multiple_show(images, window_name):
    if len(images) % 2 != 0:
        images.append(np.zeros_like(images[0]))
    print(len(images))
    row1 = np.hstack(images[:int(len(images)/2)])
    row2 = np.hstack(images[int(len(images)/2):])
    image_stack = np.vstack((row1, row2))
    cv2.imshow(window_name, image_stack)


if __name__ == '__main__':
    # TEST
    INPUT_HEIGHT = 256
    INPUT_WIDTH = 128
    K = 10
    THRESHOLD = 0.1
    QUERY = 'query_1.jpg'
    w_query = "query_images"
    w_gallery = "gallery"
    w_result = "TOP-{}_results".format(K)
    cv2.namedWindow(w_query)
    cv2.namedWindow(w_gallery)
    cv2.namedWindow(w_result)

    reid = ReIdentifier(model_name="resnet_v1_50", head_name="fc1024", model_ckpt="./market1501_weights/checkpoint-25000")
    query = prepare_image('./test_dataset/{}'.format(QUERY), INPUT_HEIGHT, INPUT_WIDTH)
    same_img_fn = os.listdir('./test_dataset/same')
    same_imgs = [prepare_image(os.path.join('./test_dataset/same', x), INPUT_HEIGHT, INPUT_WIDTH) for x in same_img_fn if x.endswith('.jpg')]
    diff_img_fn = os.listdir('./test_dataset/diff')
    diff_imgs = [prepare_image(os.path.join('./test_dataset/diff', x), INPUT_HEIGHT, INPUT_WIDTH) for x in diff_img_fn if x.endswith('.jpg')]
    gallery = [*same_imgs, *diff_imgs]
    random.shuffle(gallery)
    start = time.time()
    query_embs = reid.embed(query, 
                            input_size=[INPUT_HEIGHT, INPUT_WIDTH], 
                            augmentation=True, 
                            pre_crop_size=[144, 288])
    print('single embedding time {}'.format(time.time() - start))
    '''
    same_embs = reid.embed(same_imgs, 
                           input_size=(INPUT_HEIGHT, INPUT_WIDTH))
    diff_embs = reid.embed(diff_imgs,
                           input_size=(INPUT_HEIGHT, INPUT_WIDTH))
    same_scores = []
    diff_scores = []
    for _embs in same_embs:
        same_scores.append(reid.score(query_embs, _embs))
    for _embs in diff_embs:
        diff_scores.append(reid.score(query_embs, _embs))
    # print('dist \n same {} \n diff {}'.format(same_scores, diff_scores))
    # show_stats('same', same_scores)
    # show_stats('diff', diff_scores)
    scores = []
    for _s in same_scores:
        scores.append([_s, 1])
    for _s in diff_scores:
        scores.append([_s, 0])
    scores = np.stack(scores, axis=0)
    K = 10
    topk_idx = np.argsort(scores[:, 0])[::-1][:K]
    '''
    start = time.time()
    gallery_embs = reid.embed(gallery,
                              input_size=(INPUT_HEIGHT, INPUT_WIDTH))
    dur_time = time.time() - start
    print('{} s embedding time for {} images ({} per image)'.format(dur_time, len(gallery), dur_time / len(gallery)))
    scores = []
    for _embs in gallery_embs:
        scores.append(reid.score(query_embs, _embs))
    # topk_idx = np.argsort(scores)[::-1][:K]
    print('full list {}'.format(np.sort(scores)[::-1]))
    sorted_idx = [pick_idx for pick_idx in np.argsort(scores)[::-1] if scores[pick_idx] > THRESHOLD]
    topk_idx = sorted_idx[:min(len(sorted_idx), K)]
    scores = np.array(scores)
    print('Top {} | {}'.format(K, [scores[i] for i in topk_idx]))
    cv2.imshow(w_query, query)
    im_multiple_show(gallery, w_gallery)
    slt_result = [gallery[i] for i in topk_idx]
    if len(slt_result) <= 1:
        slt_result = [np.zeros_like(query)]
    im_multiple_show(slt_result, w_result)
    cv2.waitKey()

