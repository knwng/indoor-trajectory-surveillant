#!/usr/bin/env python3

import cv2
import os

# img_root = './annotation/caliboration/Calib_4_A1_East_Pass_Seq'
img_root = './annotation/caliboration/Calib_4_A1_South_Elevator_Seq'

def show_mouse_pos(event, x, y, flags, param):
    '''
    height, width, _ = img.shape
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, (x, 0), (x, height), 255, 2)
        cv2.line(img, (0, y), (width, y), 255, 2)
    '''
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('current pos {}'.format((x, y)))

img_list = os.listdir(img_root)
cv2.namedWindow('test')
cv2.setMouseCallback('test', show_mouse_pos)

for img_fn in img_list:
    print('process img {}'.format(img_fn))
    img = cv2.imread(os.path.join(img_root, img_fn))
    while(cv2.waitKey(33) & 0xFF != ord('q')):
        cv2.imshow('test', img)
        
'''
while (key != 113):
    cv2.imshow('test', img)
    key = cv2.waitKey(33)
    print('key {}'.format(key))
'''
