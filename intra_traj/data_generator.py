#!/usr/bin/env python2

import cv2
import re
import logging

from config import *

def logger():
    return logging.getLogger(__name__)

def bboxesGenerator(filename):
    """
    parse *_bboxes.txt
    return      : generator( (x,y,w,h) )
    """
    with open(filename,'r') as f:
        currLines = []
        currIdx   = None

        for line in f:
            pat = re.compile(r"(\d+) \| (\d+) (\d+) (\d+) (\d+)")
            (idx, x, xmax, y, ymax) = map(int, pat.match(line).groups())
            w = xmax - x
            h = ymax - y

            if not currIdx:
                currIdx = idx

            if idx == currIdx:
                currLines.append((x,y,w,h))
            else:  # idx != currIdx
                if currLines:
                    yield (currIdx, currLines)
                    currLines = []

                skippedCount = (idx-currIdx-1) // fpd
                for i in range(skippedCount):
                    yield(currIdx + (i+1)*fpd, [])

                currIdx = idx
                currLines = [(x,y,w,h)]

        if currLines:
            yield (currIdx, currLines)


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


def dataGenerator(video, bboxes):
    """
    return      : generator( (index, image, [(x,y,w,h)]) )
    """
    frames = framesGenerator(video)
    bboxes = bboxesGenerator(bboxes)
    
    (index, b) = bboxes.next()
    while True:
        (i, f) = frames.next()
        if i == index:
            break
    yield (index, f, b)

    while True:
        try:
            (i1, b) = bboxes.next()
            (i2, f) = frames.next()
        except StopIteration:
            break

        if i1 != i2:
            raise ValueError("i1 != i2")

        yield (i1, f, b)

