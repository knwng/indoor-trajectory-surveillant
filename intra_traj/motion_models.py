#!/usr/bin/env python2

from __future__ import division
import numpy as np
import logging

from filterpy.kalman import KalmanFilter

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

    def notfound(self, count):
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

    def notfound(self, count):
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

    def notfound(self, count):
        for i in range(count):
            ( x, y, w, h) = self.hist[-1]
            (dx,dy,dw,dh) = self.velocity
            self.hist.append(((x+dx),(y+dy),(w+dw),(h+dh)))
            self.velocity = tuple(map(lambda x: x*self.decay, self.velocity))
            self.counter -= 1
        return self.hist[-1] if self.counter>0 else None

    def similarity(self, frame, bbox):
        return similarityInArea(self.hist[-1], bbox)

    def extract(self):
        return [(x+w/2, y+h) for (x,y,w,h) in self.hist]


def similarityInArea((x1,y1,w1,h1), (x2,y2,w2,h2)):
    """
    return S_{overlapped} / S_{bbox1}, [0, 1], greater is better
    """
    S1 = w1 * h1

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

