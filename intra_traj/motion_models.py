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

class EMAVelocityModel(object):
    """
    """
    alpha = 0.8
    decay = 0.9

    counter = 120

    cov = np.diag([10.0, 50.0, 10.0, 50.0])

    def __init__(self, frame, (x,y,w,h), identity, timestamp):
        self.ids = [identity]
        self.counter = EMAVelocityModel.counter

        (x, y) = (x+w/2, y+h)
        self.hist = [((x,y,w,h), timestamp)]
        self.velocity = (0,0,0,0)

    def predict(self):
        (x, y, w, h) = self.hist[-1]
        (dx, dy, dw, dh) = self.velocity
        prediction = (x+dx, y+dy, w+dw, h+dh)
        return prediction

    def update(self, frame, (x,y,w,h), identity, timestamp):
        self.counter = EMAVelocityModel.counter

        if identity != self.ids[-1]:
            logger().warning("assign id {} to id {}".format(identity, self.ids[-1]))
        self.ids.append(identity)

        f = lambda new, old: self.alpha*new + (1-self.alpha)*old

        (x, y) = (x+w/2, y+h)
        (_x, _y, _w, _h) = self.hist[-1][0]
        (dx, dy, dw, dh) = (x-_x, y-_y, w-_w, h-_h)
        (_dx, _dy, _dw, _dh) = self.velocity

        updated = (f(x,_x), f(y,_y), f(w,_w), f(h,_h))
        self.hist.append((updated, timestamp))
        self.velocity = (f(dx,_dx), f(dy,_dy), f(dw,_dw), f(dh,_dh))

        return updated

    def notfound(self, frame, timestamp):
        self.hist.append((self.predict(), timestamp))
        self.velocity = tuple(map(lambda a: a*self.decay, self.velocity))
        self.counter -= 1
        return self.hist[-1][0] if self.counter>0 else None

    def similarity(self, frame, (x,y,w,h), identity):
        # probability based on multidimensional normal distribution
        position = np.transpose([(x+w/2, y+h, w, h)])
        predicted = np.transpose([self.predict()])
        cent = position - predicted
        p = exp(-1/2 * np.linalg.multi_dot([np.transpose(cent), np.linalg.inv(self.cov), cent]))
        # probability based on re-id
        q = self.ids.count(identity) / len(self.ids)
        # take both into consideration
        a = 1e-4
        probability = (1-a)*p + a*q
        logger().debug("id {}: \nxhat | x=\n{}, \np={}\nq={}\nprobability={}".format(self.ids[-1], np.hstack([predicted, position]), p, q, probability))
        return (probability, p, q)

    def extract(self):
        return [(x,y,0,0,t) for ((x,y,w,h),t) in self.hist]



class KalmanFilterModel(object):
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
    # noise at measurement
    R = np.diag([5.0, 5.0])
    # initial covariance
    P = np.diag([2.1, 2.1, 4., 4.])

    counter = 120

    def __init__(self, frame, (x,y,w,h), identity, timestamp):
        self.ids = [identity]

        self.counter = KalmanFilterModel.counter

        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix     = KalmanFilterModel.F
        self.filter.measurementMatrix    = KalmanFilterModel.H
        self.filter.processNoiseCov      = KalmanFilterModel.Q.copy()
        self.filter.measurementNoiseCov  = KalmanFilterModel.R.copy()
        self.filter.errorCovPost         = KalmanFilterModel.P.copy()
        self.filter.statePost = np.array([[x+w/2], [y+h], [0.], [0.]], dtype=np.float)

        self.hist = [(self.filter.statePost.copy(), timestamp)]

    def predict(self):
        logger().debug("id {}: |P|={}".format(self.ids[-1], np.linalg.det(self.filter.errorCovPost)))
        return self.filter.predict()

    def update(self, frame, (x,y,w,h), identity, timestamp):
        self.counter = KalmanFilterModel.counter

        if identity != self.ids[-1]:
            logger().warning("assign id {} to id {}".format(identity, self.ids[-1]))
        self.ids.append(identity)

        #self.filter.predict()
        self.filter.correct(np.array([[x+w/2], [y+h]], dtype=np.float))
        self.hist.append((self.filter.statePost.copy(), timestamp))
        return self.hist[-1][0]

    def notfound(self, frame, timestamp):
        # slow down
        decay = 0.75
        self.filter.statePost[2:] = self.filter.statePost[2:] * decay
        self.filter.statePre[2:] = self.filter.statePre[2:] * decay

        self.hist.append((self.filter.statePost.copy(), timestamp))
        self.counter -= 1
        return self.hist[-1][0] if self.counter>0 else None

    def similarity(self, frame, (x,y,w,h), identity):
        # probability based on multidimensional normal distribution
        [[_x], [_y], [_dx], [_dy]] = self.filter.statePre
        mu = self.filter.statePost
        centerized = np.array([ [x+w/2-_x], [y+h-_y], [x+w/2-_x-_dx], [y+h-_y-_dy] ], dtype=np.float)
        Sigma = self.filter.errorCovPost
        p = 1/sqrt(pow(2*pi, 4) * np.linalg.det(Sigma)) * exp(-1/2 * np.linalg.multi_dot([np.transpose(centerized), np.linalg.inv(Sigma), centerized]))
        # probability based on re-id
        q = self.ids.count(identity) / len(self.ids)
        # take both into consideration
        a = 1e-4
        probability = (1-a)*p + a*q
        logger().debug("id {}: \n|P|=\n{}, \nxhat | x=\n{}, \np={}\nq={}\nprobability={}".format(self.ids[-1], Sigma, np.hstack([mu, mu+centerized]), p, q, probability))
        #TODO: return real probability
        return (probability, p, q)

    def extract(self):
        return [(x,y,dx,dy,t) for ([[x],[y],[dx],[dy]], t) in self.hist]


