#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function

import sys
import cv2
import numpy as np
import logging
import itertools

from motion_models import MotionModel, EMAVelocityModel
from data_generator import dataGenerator

# frames per detection
fpd = 4
dt = 1/25 * fpd
index = 0  # just a global varible used in logging


def logger():
    return logging.getLogger("(index={})".format(index))

colors = itertools.cycle([(255,0,0), (0,255,0), (0,0,255)])

class Trajectories(object):
    def __init__(self, frame, bboxes, model):
        """
        frame       : image
        bboxes      : [(x,y,w,h)]
        model       : type, actually used MotionModel
        """
        self.model = model
        self.objects = [self.model(frame,b) for b in bboxes]
        logger().info("{} new objects added".format(len(self.objects)))

    def autoremove(self, index, count):
        """
        self.objects[`index`] has been un-detected for `count` frames
        remove it if gets lost
        return      : whether it is removed
        """
        position = self.objects[index].notfound(count)
        if not position:
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
        if not bboxes:
            for i in range(len(self.objects)):
                # reverse order, cause some elements may be popped
                i = len(self.objects)-i-1
                self.autoremove(i, count=1)

        elif not self.objects:
            self.objects = [self.model(frame,b) for b in bboxes]
            logger().info("{} new objects added".format(len(self.objects)))

        else:  # both bboex and self.objects are non-empty
            threshold = 0.4

            selectedModel = [-1 for _ in bboxes]
            selectedBBox  = [-1 for _ in self.objects]
            for i in range(len(self.objects)):
                sims = [self.objects[i].similarity(frame, b) for b in bboxes]
                maxJ = np.argmax(sims)
    
                if sims[maxJ] < threshold:
                    logger().debug("highest similarity for object {}: {}".format(self.objects[i].id, sims[maxJ]))
                else:
                    selectedBBox[i] = maxJ

            for j in range(len(bboxes)):
                sims = [o.similarity(frame, bboxes[j]) for o in self.objects]
                maxI = np.argmax(sims)

                if sims[maxI] < threshold:
                    logger().debug("highest similarity for bbox {}: {}".format(bboxes[j], sims[maxI]))
                else:
                    selectedModel[j] = maxI

            for j in range(len(selectedModel)):
                i = selectedModel[j]
                if i==-1 or j!=selectedBBox[i]:
                    logger().debug("bboxes {} choose object {}, which choose j={}"
                            .format(bboxes[j], "i=-1" if i==-1 else self.objects[i].id, None if i==-1 else selectedBBox[i]))
                    self.objects.append(self.model(frame, bboxes[j]))

            for i in range(len(selectedBBox)):
                # reverse order, cause some elements may be popped
                i = len(selectedBBox)-i-1
                j = selectedBBox[i]
                if j!=-1 and i==selectedModel[j]:  #
                    self.objects[i].update(frame, bboxes[j])
                elif i!=selectedModel[j]:
                    logger().debug("object {} choose bbox {}, which choose i={}"
                            .format(self.objects[i].id, bboxes[j], selectedModel[j]))
                    self.autoremove(i, count=1)
                else:  # j == -1
                    self.autoremove(i, count=1)

    def extractAll(self):
        """
        extract trajectories
        return      : [ (id, [(x,y)]) ]
        """
        return [(o.id, o.extract()) for o in self.objects]


def trajshow(image, index, trajs):
    """
    trajs       : [ (id, [(x,y)]) ]
    """
    #print("{} objects".format(len(objectsTraj)))
    for (_,traj) in trajs:  # traj: [(x,y)]
        pairs = zip(traj, traj[1:])  # pairs: [((x1,y1),(x2,y2))]
        #logger().debug("{} points".format(len(traj)))
        #logger().debug("{} lines".format(len(pairs)))
        for ((x1,y1), (x2,y2)) in pairs:
            (x1,x2,y1,y2) = map(int, (x1,x2,y1,y2))  # float -> int
            cv2.line(image, (x1,y1), (x2,y2), (255,0,0), 5)

    cv2.putText(image, str(index), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)


def bboxesshow(image, bboxes):
    for (x,y,w,h) in bboxes:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0))


def drawTrajectory(bboxesFilename, videoFilename):
    data = dataGenerator(videoFilename, bboxesFilename)
    trajs = None

    global index

    for (index, frame, bboxes) in data:
        # initialize
        if not trajs:
            trajs = Trajectories(frame, bboxes, EMAVelocityModel)
        else:
            trajs.updateAll(frame, bboxes)

        trajshow(frame, index, trajs.extractAll())
        bboxesshow(frame, bboxes)

        cv2.imshow("test", frame)
        cv2.waitKey(int(dt*800))



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
