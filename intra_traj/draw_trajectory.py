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
    def __init__(self, frame, instances, model):
        """
        frame       : image
        bboxes      : [(x,y,w,h)]
        instances   : [ DetectedInstances ]
        model       : type, derived class of MotionModel
        """
        self.model = model
        self.objects = [self.model(frame, i.bbox, i.identity) for i in instances]
        # visible[i] == True means objects[i] can be detected in the screen
        self.visible = [True for _ in self.objects]
        logger().info("{} new objects added".format(len(self.objects)))

    def autoremove(self, index, frame):
        """
        self.objects[`index`] has been un-detected for one frame
        remove it if gets lost
        return      : whether it is removed
        """
        position = self.objects[index].notfound(frame)
        if position is None:
            logger().info("objects {} lost, make it invisible".format(self.objects[index].ids[-1]))
            self.visible[index] = False
            return True
        else:
            return False

    def updateAll(self, frame, instances):
        """
        update internal motion models
        maintain trajectories info (remove old ones, add new ones)
        frame       : image
        bboxes      : [(x,y,w,h)]
        instances   : [ DetectedInstances ]
        """
        n = len(self.objects)
        k = len(instances)
        logger().debug("deciding {} instances on {} models".format(k, n))

        if k == 0:  # no bounding box found
            # reverse order, cause some elements may be popped
            for i in range(len(self.objects)-1, -1, -1):
                self.autoremove(i, frame)
            return

        elif n == 0:  # no model, init from bounding boxes
            self.objects = [self.model(frame, i.bbox, i.identity) for i in instances]
            self.visible = [True for _ in self.objects]
            logger().info("{} new objects added".format(len(self.objects)))
            return

        # both bboex and self.objects are non-empty
        threshold = 1e-60

        for i in range(n):
            if self.visible[i]:
                self.objects[i].predict()

        likelyhood = np.zeros([n, k])
        for i in range(n):
            for j in range(k):
                (probability, p, q) = self.objects[i].similarity(frame, instances[j].bbox, instances[j].identity)
                if (not self.visible[i]) and q==0:  # objects[i] has gone away, and another person comes in from the same passage
                    likelyhood[i][j] = 0
                else:
                    likelyhood[i][j] = probability if probability>threshold else threshold

        probability = likelyhood

        selectedInstance = [-1 for _ in self.objects]
        maxS = selectedInstance
        maxL = 0

        def DFS(i, selectedInstance):
            if i == n:
                ls = [likelyhood[ii][selectedInstance[ii]] for ii in range(n) if selectedInstance[ii]>=0]

                # the more -1 it have, the lower its base is
                L = pow(threshold, selectedInstance.count(-1))

                # since there's no `prod`, use `exp . sum . (map log)` instead
                L *= exp(sum(map(log, ls)))
                logger().debug("combination: {}, likelyhood: {}".format(selectedInstance, L))
                return [(selectedInstance, L)]

            else:
                result = []
                for j in range(-1, k):
                    if selectedInstance[:i].count(j)>0 and j!=-1:
                        # we don't allow a bbox selected by multiple models
#                        logger().debug("bbox {} has been selected in {}".format(j, selectedInstance[:i]))
                        continue
    
                    if j!=-1 and likelyhood[i][j]<=threshold:  # I don't want this bbox
#                        logger().debug("probability[{}][{}] <= threshold".format(i, j))
                        continue
    
                    newSelection = list(selectedInstance)  # make a copy
                    newSelection[i] = j
                    result += DFS(i+1, newSelection)
                return result


        selections = DFS(0, selectedInstance)
        maxIndex = np.argmax([l for (_,l) in selections])
        (maxS, maxL) = selections[maxIndex]

        logger().debug("final combination: {}, likelyhood: {}".format(maxS, maxL))

        for i in range(n-1, -1, -1):
            if maxS[i] == -1:
                self.autoremove(i, frame)
            else:
                self.objects[i].update(frame, instances[maxS[i]].bbox, instances[maxS[i]].identity)
                self.visible[i] = True

        for j in range(k):
            if maxS.count(j) == 0:
                self.objects.append(self.model(frame, instances[j].bbox, instances[j].identity))
                self.visible.append(True)


    def extractAll(self):
        """
        extract trajectories of all motion models
        return      : [ (id, [(x,y)]) ]
        """
        return [(o.ids[-1], o.extract()) for (v,o) in zip(self.visible, self.objects) if v]


def trajshow(image, trajs):
    """
    trajs       : [ (id, [(x,y,dx,dy)]) ]
    """
    cv2.putText(image, str(len(trajs)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    for (_,traj) in trajs:  # traj: [(x,y,dx,dy)]
        # trajectory
        traj = [(x,y) for (x,y,_,__) in traj]
        # smooth
        traj = signal.medfilt2d(traj, kernel_size=(29,1))
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


def drawTrajectory(pklFilename, videoFilename):
    data = dataGenerator(videoFilename, pklFilename)
    trajs = None

    global index

    # index     : int
    # frame     : image
    # bboxes    : [ (x,y,w,h) ]
    for (index, frame, instances) in data:
        if not trajs:  # init motion models
            trajs = Trajectories(frame, instances, KalmanFilterModel)
        else:
            trajs.updateAll(frame, instances)

        trajshow(frame, trajs.extractAll())
        bboxesshow(frame, [i.bbox for i in instances])
        # show index on top-left corner
        cv2.putText(frame, str(index), (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        cv2.imshow("trajectories", frame)
        cv2.waitKey(int(dt*100))


if __name__ == '__main__':
    videoFilename = sys.argv[1]
    pklFilename = sys.argv[2]

    logging.basicConfig(level=logging.DEBUG)

    drawTrajectory(pklFilename, videoFilename)

    cv2.destroyAllWindows()
    #for (idx, bboxes) in detectedBBoxes("single_person_bboxes.txt"):
    #    print(idx)
    #    for (x,y,w,h) in bboxes:
    #        print("  ({}, {}, {}, {})".format(x, y, w, h))
