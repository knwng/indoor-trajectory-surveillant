#!/usr/bin/env python2

from __future__ import division

import numpy as np
import logging
import cv2
from math import *
from itertools import cycle

from config import *

# frames per detection
fpd = 4
dt = 1/25 * fpd

def logger():
    return logging.getLogger(__name__)


colors = cycle([ (255,0,0)  # blue
               , (0,0,255)  # red
               , (0,255,255)  # yellow
               , (0,255,0)  # green
               , (255,0,139)  # purple
               , (0,165,255)  # orange
               , ])

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

    counter = 200

    def __init__(self, frame, (x,y,w,h), identity, timestamp):
        self.ids = [int(identity)]

        self.counter = KalmanFilterModel.counter

        self.color = next(colors)

        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.transitionMatrix     = KalmanFilterModel.F
        self.filter.measurementMatrix    = KalmanFilterModel.H
        self.filter.processNoiseCov      = KalmanFilterModel.Q.copy()
        self.filter.measurementNoiseCov  = KalmanFilterModel.R.copy()
        self.filter.errorCovPost         = KalmanFilterModel.P.copy()
        self.filter.statePost = np.array([[x+w/2], [y+h], [0.], [0.]], dtype=np.float)

        self.hist = [ [(self.filter.statePost.copy(), timestamp)] ]

    def predict(self):
        logger().debug("id {}: |P|={}".format(self.ids[-1], np.linalg.det(self.filter.errorCovPost)))
        return self.filter.predict()

    def update(self, frame, (x,y,w,h), identity, timestamp):
        identity = int(identity)

        self.counter = KalmanFilterModel.counter

        if identity != self.ids[-1]:
            logger().warning("assign id {} to id {}".format(identity, self.ids[-1]))
        self.ids.append(identity)

        #self.filter.predict()
        self.filter.correct(np.array([[x+w/2], [y+h]], dtype=np.float))
        self.hist[-1].append((self.filter.statePost.copy(), timestamp))
        return self.hist[-1][-1][0]

    def notfound(self, frame, timestamp):
        # slow down
        decay = 0.75
        self.filter.statePost[2:] = self.filter.statePost[2:] * decay
        self.filter.statePre[2:] = self.filter.statePre[2:] * decay

        self.hist[-1].append((self.filter.statePost.copy(), timestamp))
        self.counter -= 1
        return self.hist[-1][-1][0] if self.counter>0 else None

    def similarity(self, frame, (x,y,w,h), identities):
        # probability based on multidimensional normal distribution
        [[_x], [_y], [_dx], [_dy]] = self.filter.statePre
        mu = self.filter.statePost
        centerized = np.array([ [x+w/2-_x], [y+h-_y], [x+w/2-_x-_dx], [y+h-_y-_dy] ], dtype=np.float)
        Sigma = self.filter.errorCovPost
        p = 1/sqrt(pow(2*pi, 4) * np.linalg.det(Sigma)) * exp(-1/2 * np.linalg.multi_dot([np.transpose(centerized), np.linalg.inv(Sigma), centerized]))
        # probability based on re-id
        q = 0
        for (identity, score) in identities:
            q += score * self.ids.count(int(identity)) / len(self.ids)
        # take both into consideration
        a = 1e-4
        probability = (1-a)*p + a*q
        logger().debug("id {}: \n|P|=\n{}, \nxhat | x=\n{}, \np={}\nq={}\nprobability={}".format(self.ids[-1], Sigma, np.hstack([mu, mu+centerized]), p, q, probability))
        #TODO: return real probability
        return (probability, p, q)

    def extract(self):
        logger().debug("self.hist has {} part(s)".format(len(self.hist)))
        return [[(x,y,dx,dy,t) for ([[x],[y],[dx],[dy]], t) in trajectory] for trajectory in self.hist]


