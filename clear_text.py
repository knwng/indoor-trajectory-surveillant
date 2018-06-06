#!/usr/bin/env python3

import cv2
import numpy as np
import os

img_root = './annotation/caliboration/Calib_4_A1_South_Elevator_Seq/'
output_root = './annotation/caliboration/Calib_4_A1_South_Elevator_Seq_cgd'
if not os.path.exists(output_root):
    os.makedirs(output_root)

img_list = os.listdir(img_root)

for img_fn in img_list:
    print('processing {}'.format(img_fn))
    img = cv2.imread(os.path.join(img_root, img_fn))
    # cv2.drawRectangle(img, (28, 71), (749, 105), (0, 0, 0), 2)
    # cv2.drawRectangle(img, (978, 789), (1289, 838), (0, 0, 0), 2)
    img[767:807, 948:1303, :] = 255
    img[64:114, 28:749, :] = 255
    cv2.imwrite(os.path.join(output_root, img_fn), img)

