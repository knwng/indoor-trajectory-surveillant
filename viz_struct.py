#!/usr/bin/env python3
from surveillant import *
import pickle
import cv2

def display_struct():
    data = pickle.load(open('output/struct_storage/0509_1_A3_ELEVATOR_slice.pkl', 'rb'))

    for _frame in data['frames']:
        if _frame.identity is None:
            continue
        print('timestamp {} | bbox {} | id {} | idc {}'.format(_frame.timestamp, _frame.bbox, _frame.identity, _frame.id_candidates))
    print('video dir {}'.format(data['video_fn']))
    return data

def viz_struct():
    data = pickle.load(open('output/struct_storage/0509_1_A3_ELEVATOR_slice.pkl', 'rb'))
    cap = cv2.VideoCapture(data['video_fn'])
    frame_idx = 0
    while cap.isOpened():
        frame_idx += 1
        if frame_idx % data['detect_interval'] != 0:
            continue
        ret, frame = cap.read()
        if frame is None:
            print('No stream')
            break

    pass
    
if __name__ == '__main__':
    display_struct()
