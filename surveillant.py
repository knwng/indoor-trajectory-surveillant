#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os, sys
import cv2 
import json

sys.path.insert(0, './reid')
from reid import ReIdentifier
sys.path.insert(0, './detector')
from detector import Detector

model_root = './cv_models'

class Surveillant(object):
    
    def __init__(self, detector_ckpt, reid_ckpt, reid_model_name="resnet_v1_50", reid_head_name="fc1024"):
        self.detector = Detector(ckpt=os.path.join(os.getcwd(), detector_ckpt))
        self.reid = ReIdentifier(model_name=reid_model_name, 
                                 head_name=reid_head_name,
                                 model_ckpt=os.path.join(os.getcwd(), reid_ckpt),
                                 input_height=256,
                                 input_width=128,
                                 surveillant_map=None)



    def detect(self, img, timestamp):
        pass

if __name__ == '__main__':
    img = cv2.imread("./test.jpg")
    surve = Surveillant(detector_ckpt='cv_models/ssd_300_vgg.ckpt',
                        reid_ckpt='cv_models/checkpoint-25000')
                        # reid_ckpt='cv_models/reid_ckpt-25000')
    surve.detector.detect(img, viz=True)
    cv2.waitKey()


