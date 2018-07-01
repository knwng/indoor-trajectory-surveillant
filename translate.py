#!/usr/bin/env python3

# since py2 cannot import py3 modules,
# this file is written to translate data from upstream

import os
import sys
import pickle
import cv2
from structs import *

skip = 0

def frameInstancesGenerator(pklFilename):
    with open(pklFilename, 'rb') as f:
        pkl = pickle.load(f)

    videoFilename = os.path.join(os.path.dirname(pklFilename), pkl['video_fn'])
    capture = cv2.VideoCapture(videoFilename)
    if not capture.isOpened():
        raise ValueError("file {} does not exist".format(filename))

    timestamp = int(pkl['timestamp_offset'])
    instances = (timestamp, [])

    allInstances = iter(pkl['frames'])

    EOF = False
    while not EOF:
        timestamp += 1 + skip

        # seek the frame
        for i in range(skip+1):
            (ok, frame) = capture.read()
            # memory usage too huge
            frame = None
            if not ok:
                EOF = True
                break

        if EOF:
            break

        if instances[1]!=[] and instances[0] != timestamp:
            yield FrameInstances(timestamp, frame, [])
            continue

        while True:
            try:
                inst = next(allInstances)
            except StopIteration:
                EOF = True
                break

            if inst.timestamp == timestamp:
                instances[1].append(DetectedInstance(inst.cropped_img_url, bboxConvert(inst.bbox), inst.identity, inst.id_candidates, inst.embedding))
            else:
                yield FrameInstances(timestamp, frame, instances[1])
                instances = (inst.timestamp, [DetectedInstance(inst.cropped_img_url, bboxConvert(inst.bbox), inst.identity, inst.id_candidates, inst.embedding)])
                break


def bboxConvert(bbox):
    (xmin, xmax, ymin, ymax) = bbox
    (x, y, w, h) = (xmin, ymin, xmax-xmin, ymax-ymin)
    return (x, y, w, h)

if __name__ == '__main__':
    pklFilename = sys.argv[1]
    newPKL = os.path.splitext(pklFilename)[0] + "_translated.pkl"

    translated = list(frameInstancesGenerator(pklFilename))
    with open(newPKL, 'wb') as f:
        pickle.dump(translated, f, protocol=2, fix_imports=True)
