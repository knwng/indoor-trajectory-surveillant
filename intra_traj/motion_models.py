#!/usr/bin/env python2

from __future__ import division

import numpy as np
import logging
import cv2
from math import *

from config import *

# frames per detection
fpd = 4
dt = 1/25 * fpd

def logger():
    return logging.getLogger(__name__)


class MotionModel(object):
    id = 0
    counter = 15

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
        self.counter = MotionModel.counter
        self.counter -= 1

    def update(self, frame, bbox):
        """
        frame       : image
        bbox        : (x,y,w,h)
        return      : (x,y,w,h)
        """
        raise NotImplementedError

    def notfound(self, frame):
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


class KalmanFilterModel(MotionModel):
    """
    Kalman filter based
    state: x, y, dx, dy
    """
    # transition matrix
    F = np.array([ [ 1., 0., 1., 0. ]
                 , [ 0., 1., 0., 1. ]
                 , [ 0., 0., 1., 0. ]
                 , [ 0., 0., 0., 1. ]
                 ])
    # measurement matrix
    H = np.eye(2, 4)
    # noise at transition
    Q = np.diag([1e0, 1e0, 1e1, 1e1])
    #Q = 1e-5 * np.eye(4)
    # noise at measurement
    R = np.diag([5.0, 5.0])
    #R = 0.1 * np.eye(2)
    # initial covariance
    P = np.diag([2.1, 2.1, 4., 4.])
    #P = 0.1 * np.eye(4)

    def __init__(self, frame, (x,y,w,h)):
        MotionModel.__init__(self, frame, (x,y,w,h))

        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix     = KalmanFilterModel.F
        self.filter.measurementMatrix    = KalmanFilterModel.H
        self.filter.processNoiseCov      = KalmanFilterModel.Q.copy()
        self.filter.measurementNoiseCov  = KalmanFilterModel.R.copy()
        self.filter.errorCovPost         = KalmanFilterModel.P.copy()
        self.filter.statePost = np.array([[x+w/2], [y+h], [0.], [0.]])

        self.hist = [self.filter.statePost.copy()]

    def predict(self):
        logger().debug("id {}: |P|={}".format(self.id, np.linalg.det(self.filter.errorCovPost)))
        return self.filter.predict()

    def update(self, frame, (x,y,w,h)):
        self.counter = MotionModel.counter
        self.counter -= 1
        #self.filter.predict()
        self.filter.correct(np.array([[x+w/2], [y+h]], dtype=np.float))
        self.hist.append(self.filter.statePost.copy())
        return self.hist[-1]

    def notfound(self, frame=None):
        self.hist.append(self.filter.statePost.copy())
        self.counter -= 1
        return self.hist[-1] if self.counter>0 else None

    def _similarity(self, frame, (x,y,w,h)):
        mu = self.filter.statePost[0:2]
        X = np.array([[x+w/2], [y+h]], dtype=np.float)
        Sigma = self.filter.errorCovPost[0:2,0:2]
        probability = 1/sqrt(pow(2*pi, 4) * np.linalg.det(Sigma)) * exp(-1/2 * np.linalg.multi_dot([np.transpose(X-mu), np.linalg.inv(Sigma), X-mu]))
        logger().debug("id {}: \n|Sigma|=\n{}, \nxhat | x=\n{}, \np={}".format(self.id, Sigma, np.hstack([mu, X]), probability))
        return probability

    def similarity(self, frame, (x,y,w,h)):
        [[_x], [_y], [_dx], [_dy]] = self.filter.statePre
        mu = self.filter.statePost
        centerized = np.array([ [x+w/2-_x], [y+h-_y], [x+w/2-_x-_dx], [y+h-_y-_dy] ], dtype=np.float)
        Sigma = self.filter.errorCovPost
        probability = 1/sqrt(pow(2*pi, 4) * np.linalg.det(Sigma)) * exp(-1/2 * np.linalg.multi_dot([np.transpose(centerized), np.linalg.inv(Sigma), centerized]))
        logger().debug("id {}: \n|P|=\n{}, \nxhat | x=\n{}, \np={}".format(self.id, Sigma, np.hstack([mu, mu+centerized]), probability))
        return probability

    def extract(self):
        return [(x,y,dx,dy) for [[x],[y],[dx],[dy]] in self.hist]

class EMAVelocityModel(MotionModel):
    """
    exponential moving average (on velocity), a naive way
    """
    alpha = 0.8
    decay = 0.9

    def __init__(self, frame, (x,y,w,h)):
        MotionModel.__init__(self, frame, (x,y,w,h))
        self.hist = [(x+w/2, y+h, w, h)]
        self.velocity = (0,0,0,0)

    def predict(self):
        pass

    def update(self, frame, (x,y,w,h)):
        (x,y) = (x+w/2, y+h)

        f = lambda new, old: self.alpha*new + (1-self.alpha)*old

        (x_,y_,w_,h_) = self.hist[-1]
        (dx,dy,dw,dh) = (x-x_,y-y_,w-w_,h-h_)
        (dx_,dy_,dw_,dh_) = self.velocity

        updated = (f(x,x_), f(y,y_), f(w,w_), f(h,h_))
        self.hist.append((f(x,x_), f(y,y_), f(w,w_), f(h,h_)))
        self.velocity = (f(dx,dx_), f(dy,dy_), f(dw,dw_), f(dh,dh_))
        return updated

    def notfound(self, frame=None):
        ( x, y, w, h) = self.hist[-1]
        (dx,dy,dw,dh) = self.velocity
        self.hist.append(((x+dx),(y+dy),(w+dw),(h+dh)))
        self.velocity = tuple(map(lambda x: x*self.decay, self.velocity))
        self.counter -= 1
        return self.hist[-1] if self.counter>0 else None

    def similarity(self, frame, (x,y,w,h)):
        (x, y) = (x+w/2, y+h)
        ( _x, _y, _w, _h) = self.hist[-1]
        (_dx,_dy,_dw,_dh) = self.velocity
        Sigma = _w*_h/20000 * np.diag([1, 1, 1, 1])
        c = [x-_x-_dx,y-_y-_dy,w-_w-_dw,h-_h-_dh]
        probability = 1/sqrt(pow(2*pi, 4) * np.linalg.det(Sigma)) * exp(-1/2 * np.linalg.multi_dot([c, np.linalg.inv(Sigma), c]))
        logger().debug("id {}: \nxhat | x=\n{}, \np={}".format(self.id, np.array([self.hist[-1], (x,y,w,h)]).transpose(), probability))
        return probability

    def extract(self):
        traj = [(x, y, 0, 0) for (x,y,w,h) in self.hist]
        traj[-1][2:] = self.velocity[:2]
        return traj

