#!/usr/bin/env python2


from __future__ import division
from __future__ import print_function 

import sys
import cv2
import numpy as np
import logging
import itertools
from scipy import signal

from data_generator import dataGenerator
from motion_models import *
from config import *

# frames per detection
index = 0  # just a global varible used in logging


def logger():
    return logging.getLogger("(index={})".format(index))

colors = itertools.cycle([(255,0,0), (0,255,0), (0,0,255)])


class Trajectories(object):
    def __init__(self, frame, bboxes, model):
        """
        frame       : image
        bboxes      : [(x,y,w,h)]
        model       : type, derived class of MotionModel
        """
        self.model = model
        self.objects = [self.model(frame,b) for b in bboxes]
        logger().info("{} new objects added".format(len(self.objects)))

    def autoremove(self, index, frame):
        """
        self.objects[`index`] has been un-detected for one frame
        remove it if gets lost
        return      : whether it is removed
        """
        position = self.objects[index].notfound(frame)
        if position is None:
            logger().info("objects {} lost, removed".format(self.objects[index].id))
            self.objects.pop(index)
            return True
        else:
            return False

    def updateAll(self, frame, bboxes):
        """
        update internal motion models
        maintain trajectories info (remove old ones, add new ones)
        frame       : image
        bboxes      : [(x,y,w,h)]
        """
        n = len(self.objects)
        k = len(bboxes)
        logger().debug("deciding {} bboxes on {} models".format(k, n))

        if k == 0:  # no bounding box found
            # reverse order, cause some elements may be popped
            for i in range(len(self.objects)-1, -1, -1):
                self.autoremove(i, frame)
            return

        elif n == 0:  # no model, init from bounding boxes
            self.objects = [self.model(frame,b) for b in bboxes]
            logger().info("{} new objects added".format(len(self.objects)))
            return

        # both bboex and self.objects are non-empty
        threshold = 1e-60

        for i in range(n):
            self.objects[i].predict()

        likelyhood = np.zeros([n, k])
        for i in range(n):
            for j in range(k):
                lh = self.objects[i].similarity(frame, bboxes[j])
                likelyhood[i][j] = lh if lh>threshold else threshold

        probability = likelyhood

        selectedBBox = [-1 for _ in self.objects]
        maxS = selectedBBox
        maxL = 0

        def DFS(i, selectedBBox):
            if i == n:
                ls = [likelyhood[ii][selectedBBox[ii]] for ii in range(n) if selectedBBox[ii]>=0]

                # the more -1 it have, the lower its base is
                L = pow(threshold, selectedBBox.count(-1))

                # since there's no `prod`, use `exp . sum . (map log)` instead
                L *= exp(sum(map(log, ls)))
                logger().debug("combination: {}, likelyhood: {}".format(selectedBBox, L))
                return [(selectedBBox, L)]

            else:
                result = []
                for j in range(-1, k):
                    if selectedBBox[:i].count(j)>0 and j!=-1:
                        # we don't allow a bbox selected by multiple models
#                        logger().debug("bbox {} has been selected in {}".format(j, selectedBBox[:i]))
                        continue
    
                    if j!=-1 and likelyhood[i][j]<=threshold:  # I don't want this bbox
#                        logger().debug("probability[{}][{}] <= threshold".format(i, j))
                        continue
    
                    newSelection = list(selectedBBox)  # make a copy
                    newSelection[i] = j
                    result += DFS(i+1, newSelection)
                return result


        selections = DFS(0, selectedBBox)
        maxIndex = np.argmax([l for (_,l) in selections])
        (maxS, maxL) = selections[maxIndex]

        logger().debug("final combination: {}, likelyhood: {}".format(maxS, maxL))

        for i in range(n-1, -1, -1):
            if maxS[i] == -1:
                self.autoremove(i, frame)
            else:
                self.objects[i].update(frame, bboxes[maxS[i]])

        for j in range(k):
            if maxS.count(j) == 0:
                self.objects.append(self.model(frame, bboxes[j]))


    def extractAll(self):
        """
        extract trajectories of all motion models
        return      : [ (id, [(x,y)]) ]
        """
        return [(o.id, o.extract()) for o in self.objects]

def trajshow(image, trajs):
    """
    trajs       : [ (id, [(x,y,dx,dy)]) ]
    """
    cv2.putText(image, str(len(trajs)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    for (_,traj) in trajs:  # traj: [(x,y,dx,dy)]
        # trajectory
        traj = [(x,y) for (x,y,_,__) in traj]
        # smooth
        traj = signal.medfilt2d(traj, kernel_size=(3,1))
        pairs = zip(traj, traj[1:])  # pairs: [((x1,y1),(x2,y2))]
        for ((x1,y1), (x2,y2)) in pairs:
            (x1,x2,y1,y2) = map(int, (x1,x2,y1,y2))  # float -> int
            cv2.line(image, (x1,y1), (x2,y2), (255,0,0), 4)


def _trajshow(image, trajs):
    """
    trajs       : [ (id, [(x,y,dx,dy)]) ]
    """
    #print("{} objects".format(len(objectsTraj)))
    maxVelocity = 10
    cv2.putText(image, str(len(trajs)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    for (_,traj) in trajs:  # traj: [(x,y,dx,dy)]
        # velocity
        (x,y,dx,dy) = map(int, traj[-1])
        #if np.linalg.norm([dx, dy]) < maxVelocity:
        if True:
            cv2.arrowedLine(image, (x,y), (x+8*dx,y+8*dy), (0,0,255), 2)
            # current location
            cv2.circle(image, (x,y), 10, (255,0,0), 1)

            # trajectory
            pairs = zip(traj, traj[1:])  # pairs: [((x1,y1,dx1,dy1),(x2,y2,dx2,dy2))]
#            logger().debug(str(pairs))
            for ((x1,y1,dx1,dy1), (x2,y2,dx2,dy2)) in pairs:
                (x1,x2,y1,y2) = map(int, (x1,x2,y1,y2))  # float -> int
                cv2.line(image, (x1,y1), (x2,y2), (255,0,0), 4)


def bboxesshow(image, bboxes):
    for (x,y,w,h) in bboxes:
        (x,y,w,h) = (x,y,w,h)
        (x,y,w,h) = map(int, (x,y,w,h))
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0))


def drawTrajectory(bboxesFilename, videoFilename):
    data = dataGenerator(videoFilename, bboxesFilename)
    trajs = None

    global index

    # index     : int
    # frame     : image
    # bboxes    : [ (x,y,w,h) ]
    for (index, frame, bboxes) in data:
        if not trajs:  # init motion models
            trajs = Trajectories(frame, bboxes, KalmanFilterModel)
        else:
            trajs.updateAll(frame, bboxes)

        trajshow(frame, trajs.extractAll())
        bboxesshow(frame, bboxes)
        # show index on top-left corner
        cv2.putText(frame, str(index), (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        cv2.imshow("trajectories", frame)
        cv2.waitKey(int(dt*1200))


if __name__ == '__main__':
    videoFilename = sys.argv[1]
    bboxesFilename = sys.argv[2]
    logging.basicConfig(level=logging.DEBUG)
    drawTrajectory(bboxesFilename, videoFilename)
    cv2.destroyAllWindows()
    #for (idx, bboxes) in detectedBBoxes("single_person_bboxes.txt"):
    #    print(idx)
    #    for (x,y,w,h) in bboxes:
    #        print("  ({}, {}, {}, {})".format(x, y, w, h))
