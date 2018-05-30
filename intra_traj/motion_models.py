#!/usr/bin/env python2

from __future__ import division
import numpy as np
import logging
import cv2
from math import log

from filterpy.kalman import KalmanFilter

from config import *

# frames per detection
fpd = 4
dt = 1/25 * fpd

def logger():
    return logging.getLogger(__name__)


class MotionModel(object):
    id = 0

    def __init__(self, frame, bbox):
        """
        frame       : image
        bbox        : (x,y,w,h)
        """
        # ensure each class instance has a distinct id
        self.id = MotionModel.id
        self.id += 1
        MotionModel.id += 1

        # lost countdown
        self.counter = 15

    def update(self, frame, bbox):
        """
        frame       : image
        bbox        : (x,y,w,h)
        return      : (x,y,w,h)
        """
        raise NotImplementedError

    def notfound(self, count, frames):
        """
        count       : how many times it has been un-detectable
        return      : (x,y,w,h) if still_in_screen else None
        """
        raise NotImplementedError

    def similarity(self, frame, bbox):
        """
        between NEXT state and bbox
        return      : greater is more similar
        """
        raise NotImplementedError

    def extract(self):
        """
        obtain its trajectory, past to present
        return      : [(x,y)]
        """
        raise NotImplementedError


class NaiveModel(MotionModel):
    """
    use currently detected bbox, just for test
    """
    def __init__(self, frame, bbox):
        MotionModel.__init__(self, frame, bbox)
        self.hist = [bbox]

    def update(self, frame, bbox):
        self.hist.append(bbox)
        return updated

    def notfound(self, count, frames=None):
        for i in range(count):
            self.counter -= 1
        return self.hist[-1] if self.counter>0 else None

    def similarity(self, frame, bbox):
        return similarityInArea(self.hist[-1], bbox)

    def extract(self):
        return [(x+w/2, y+h) for (x,y,w,h) in self.hist]


class EMAVelocityModel(MotionModel):
    """
    exponential moving average (on velocity), a naive way
    """
    alpha = 0.8
    decay = 0.9

    def __init__(self, frame, bbox):
        MotionModel.__init__(self, frame, bbox)
        self.hist = [bbox]
        self.velocity = (0,0,0,0)

    def update(self, frame, (x,y,w,h)):
        f = lambda new, old: self.alpha*new + (1-self.alpha)*old

        (x_,y_,w_,h_) = self.hist[-1]
        (dx,dy,dw,dh) = (x-x_,y-y_,w-w_,h-h_)
        (dx_,dy_,dw_,dh_) = self.velocity

        updated = (f(x,x_), f(y,y_), f(w,w_), f(h,h_))
        self.hist.append((f(x,x_), f(y,y_), f(w,w_), f(h,h_)))
        self.velocity = (f(dx,dx_), f(dy,dy_), f(dw,dw_), f(dh,dh_))
        return updated

    def notfound(self, count, frames=None):
        for i in range(count):
            ( x, y, w, h) = self.hist[-1]
            (dx,dy,dw,dh) = self.velocity
            self.hist.append(((x+dx),(y+dy),(w+dw),(h+dh)))
            self.velocity = tuple(map(lambda x: x*self.decay, self.velocity))
            self.counter -= 1
        return self.hist[-1] if self.counter>0 else None

    def similarity(self, frame, bbox):
        return similarityInArea(self.hist[-1], bbox)
        #return crossEntropy(self.hist[-1], bbox)

    def extract(self):
        return [(x+w/2, y+h) for (x,y,w,h) in self.hist]


class KalmanFilterModel(MotionModel):
    """
    Kalman filter based
    state: x, y, dx, dy
    """
    # transition matrix
    F = np.array([ [ 1., 0., dt, 0. ]
                 , [ 0., 1., 0., dt ]
                 , [ 0., 0., 1., 0. ]
                 , [ 0., 0., 0., 1. ]
                 ])
    # measurement matrix
    H = np.eye(2, 4)
    # noise at transition
    Q = np.diag([10., 10., 10., 10.])
    # noise at measurement
    R = np.diag([1., 1.])
    # initial covariance
    P = np.diag([1., 1., 20., 20.])

    def __init__(self, frame, (x,y,w,h)):
        MotionModel.__init__(self, frame, (x,y,w,h))
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix     = F
        self.filter.measurementMatrix    = H
        self.filter.processNoiseCov      = Q
        self.filter.measurementNoiseCov  = R
        self.filter.errorCovPost         = P
        self.filter.statePost = np.array([[x], [y], [0.], [0.]])
        self.hist = [self.filter.statePost]

    def update(self, frame, (x,y,w,h)):
        self.filter.predict()
        self.filter.correct(np.array([[x], [y], [0.], [0.]]))
        self.hist.append(self.filter.statePost)
        return self.hist[-1]

    def notfound(self, count, frames=None):
        for i in range(count):
            self.predict()
            self.hist.append(self.filter.statePost)
            self.count -= 1
        return self.hist[-1] if self.count>0 else None

    def similarity(self, frame, (x,y,w,h)):
        pass

    def extract(self):
        pass


class TrackingModel(MotionModel):
    """
    model using OpenCV tracking API
    """
    create = cv2.TrackerKCF_create

    def __init__(self, frame, bbox):
        MotionModel.__init__(self, frame, bbox)
        self.tracker = TrackingModel.create()
        self.tracker.init(frame, bbox)
        self.hist = [bbox]

    def update(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self.hist.append(bbox)
        return bbox

    def notfound(self, count, frames=None):
        for i in range(count):
            frame = frames[i]
            (ok, bbox) = self.tracker.update(frame)
            self.hist.append(bbox)
            self.counter -= 1
        return self.hist[-1] if self.counter>0 else None

    def similarity(self, frame, bbox):
        #return similarityInArea(self.hist[-1], bbox)
        return crossEntropy(self.hist[-1], bbox)

    def extract(self):
        return [(x+w/2, y+h) for (x,y,w,h) in self.hist]


def crossEntropy((x1,y1,w1,h1), (x2,y2,w2,h2)):
    """
    1: estimated, 2: actual
    """
    p = 1 / w2 / h2
    q = 1 / w1 / h1
    result = 0
    for i in range(w2):
        for j in range(h2):
            point = (x2+i, y2+j)
            if x1 <= point[0] <= x1+w1 and y1 <= point[1] <= y1+h1:
                result -= p * log(q)
            else:
                pass
    logger().debug("crossEntropy: {}".format(result))
    return result


def similarityInArea((x1,y1,w1,h1), (x2,y2,w2,h2)):
    """
    return S_{overlapped} / S_{bbox1}, [0, 1], greater is better
    """
    S1 = w1 * h1
    if S1 == 0:
        return int(x1 < x2 < x1+w1 and y1 < y2 < y1+h1)

    def clamp(x, minVal, maxVal):
        return sorted((x, minVal, maxVal))[1]

    if x2 < x1:
        w12 = clamp(x2+w2-x1, 0, w1)
    else:
        w12 = clamp(x1+w1-x2, 0, w2)

    if y2 < y1:
        h12 = clamp(y2+h2-y1, 0, h1)
    else:
        h12 = clamp(y1+h1-y2, 0, h2)

    S12 = w12 * h12
    return S12 / S1

