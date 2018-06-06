#!/usr/bin/env python3

import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--file', required=True)

args = ap.parse_args()

def show_mouse_pos(event, x, y, flags, param):
    '''
    height, width, _ = img.shape
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, (x, 0), (x, height), 255, 2)
        cv2.line(img, (0, y), (width, y), 255, 2)
    '''
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('current pos {}'.format((x, y)))

if not os.path.isfile(args.file):
    print('{} is not a file'.format(args.file))
    raise FileNotFoundError

cv2.namedWindow('test')
cv2.setMouseCallback('test', show_mouse_pos)

img = cv2.imread(args.file)
while(cv2.waitKey(33) & 0xFF != ord('q')):
    cv2.imshow('test', img)
