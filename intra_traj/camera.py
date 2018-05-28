#!/usr/bin/env python2

import numpy as np
import cv2
from scipy.io import loadmat

class cameraParameters(object):
    """
    similar to cameraParameters in MATLAB
    """
    def __init__(self, matfile):
        params = loadmat(matfile)
        self.intrinsicMat = params['IntrinsicMatrix']
        self.radDist= params['RadialDistortion']
        self.tanDist= params['TangentialDistortion']
        #self.rotMats = params['RotationMatrices']
        #self.rotVecs = params['RotationVectors']
        #self.transVecs = params['TranslationVectors']

    def undistort(self, image):
        cameraMatrix = self.intrinsicMat.transpose()
        [[k1,k2,k3]] = self.radDist
        [[p1,p2]] = self.tanDist
        distCoeffs = np.array([k1,k2,p1,p2,k3])
        return cv2.undistort(image, cameraMatrix, distCoeffs)
