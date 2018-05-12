#!/usr/bin/env python3

from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import loss
from loss import cdist
import os
import cv2

import h5py
import json
import numpy as np
import tensorflow as tf

from aggregators import AGGREGATORS
import common

INPUT_HEIGHT = 256
INPUT_WIDTH = 128

def flip_augment(image, fid, pid):
    """ Returns both the original and the horizontal flip of an image. """
    images = tf.stack([image, tf.reverse(image, [1])])
    return images, [fid]*2, [pid]*2


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

    def __init__(self, model_name, head_name, model_ckpt, surveillant_map=None):
        model = import_module('nets.' + model_name)
        head = import_module('heads.' + head_name)

        self.image = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_WIDTH, INPUT_HEIGHT, 3], name='image')

        self.endpoints, body_prefix = model.endpoints(self.image, is_training=False)
        with tf.name_scope('head'):
            self.endpoints = head.head(self.endpoints, 128, is_training=False)
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, model_ckpt)
        self.surveillant_map = surveillant_map

    def embed(self, image):
        emb = self.sess.run(self.endpoints['emb'], feed_dict={self.image:image})
        return emb

    def dist(self, ftr1, ftr2, single_batch=True, metric='euclidean'):
        if single_batch:
            return np.sqrt(np.sum((ftr1 - ftr2) ** 2))
        else:
            raise NotImplementedError


def prepare_image(img_dir, height, width):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (height, width))
    return img

if __name__ == '__main__':
    # TEST
    reid = ReIdentifier(model_name="resnet_v1_50", head_name="fc1024", model_ckpt="./market1501_weights/checkpoint-25000")
    img_dirs = ['./query.jpg', './same.jpg', './diff.jpg']
    imgs = [prepare_image(_dir, INPUT_HEIGHT, INPUT_WIDTH) for _dir in img_dirs]
    embs = reid.embed(imgs)
    dist_same = reid.dist(embs[0], embs[1])
    dist_diff = reid.dist(embs[0], embs[2])
    print('dist | same {} | diff {}'.format(dist_same, dist_diff))
