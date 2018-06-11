#!/usr/bin/env python2

import cv2
import re
import logging
import cPickle as pickle

from config import *

def logger():
    return logging.getLogger(__name__)


def bboxesGenerator(filename):
    """
    return      : generator( FrameInstances )
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return iter(data)

    ## image has been multiplified
    #for f in data:
    #    for i in f.instances:
    #        i.bbox = i.bbox / 2
    #    yield f


def framesGenerator(filename):
    """
    read video frame by frame
    return      : generator( image )
    """
    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        raise ValueError("file {} does not exist".format(filename))
    index = 0

    while True:
        (ok, frame) = capture.read()
        if not ok:
            break

        index += 1
        if index % fpd == 0:
            yield (index, frame)


def dataGenerator(video, pkl):
    """
    return      : generator( (index, image, [(x,y,w,h)]) )
    """
    frames = framesGenerator(video)
    bboxes = bboxesGenerator(pkl)
    
    instances = bboxes.next()
    while True:
        (i, f) = frames.next()
        if i == instances.timestamp:
            break
    yield (i, f, instances.instances)

    while True:
        try:
            instances = bboxes.next()
            (i, f) = frames.next()
        except StopIteration:
            break

        if instances.timestamp != i:
            raise ValueError("i1 != i2")

        yield (i, f, instances.instances)

