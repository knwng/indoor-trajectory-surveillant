#!/usr/bin/env python2


from __future__ import division
from __future__ import print_function 

import os
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
    def __init__(self, frame, instances, model, timestamp):
        """
        frame       : image
        bboxes      : [(x,y,w,h)]
        instances   : [ DetectedInstances ]
        model       : type, derived class of MotionModel
        """
        self.model = model
        self.objects = [self.model(frame, i.bbox, i.identity, timestamp) for i in instances]
        # visible[i] == True means objects[i] can be detected in the screen
        self.visible = [True for _ in self.objects]
        logger().info("{} new objects added".format(len(self.objects)))

    def autoremove(self, index, frame, timestamp):
        """
        self.objects[`index`] has been un-detected for one frame
        remove it if gets lost
        return      : whether it is removed
        """
        position = self.objects[index].notfound(frame, timestamp)
        if position is None:
            logger().info("objects {} lost, make it invisible".format(self.objects[index].ids[-1]))
            self.visible[index] = False
            self.objects[index].hist.append([])
            logger().debug("object {} has {} part(s) after truncation".format(index, len(self.objects[index].hist)))
            if len([1 for traj in self.objects[index].hist for p in traj]) < 10:
                logger().warning("objects {}'s history is too short, removing it".format(self.objects[index].ids[-1]))
                self.objects.pop(index)
                self.visible.pop(index)
            return True
        else:
            return False

    def updateAll(self, frame, instances, timestamp):
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
                if self.visible[i]:
                    self.autoremove(i, frame, timestamp)
            return

        elif n == 0:  # no model, init from bounding boxes
            self.objects = [self.model(frame, i.bbox, i.identity, timestamp) for i in instances]
            self.visible = [True for _ in self.objects]
            logger().info("{} new objects added".format(len(self.objects)))
            return

        # both bboex and self.objects are non-empty
        threshold = 1e-60

        if self.model is KalmanFilterModel:
            for i in range(n):
                if self.visible[i]:
                    self.objects[i].predict()

        likelyhood = np.zeros([n, k])
        for i in range(n):
            for j in range(k):
                (probability, p, q) = self.objects[i].similarity(frame, instances[j].bbox, instances[j].id_candidates)
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
                if self.visible[i]:
                    self.autoremove(i, frame, timestamp)
            else:  # maxS[i] != -1
                self.objects[i].update(frame, instances[maxS[i]].bbox, instances[maxS[i]].identity, timestamp)
                self.visible[i] = True

        # add new models
        for j in range(k):
            if maxS.count(j) == 0:
                self.objects.append(self.model(frame, instances[j].bbox, instances[j].identity, timestamp))
                self.visible.append(True)
                logger().debug("add new model {}".format(instances[j].identity))


    def extractAll(self):
        """
        extract trajectories of all motion models
        return      : [ (id, [(x,y)]) ]
        """
        return [(o.color, o.extract()) for (v,o) in zip(self.visible, self.objects) if v]


def distance((x1,y1), (x2,y2)):
    """
    simple function to calculate \sqrt{||p1-p2||^2}
    """
    dist = np.linalg.norm([x1-x2, y1-y2])
    logger().debug("distance: {}".format(dist))
    return dist


def forbidden((x,y)):
    """
    checks whether (x,y) on image plane locates in forbidden zone
    """
    (newX, newY) = projectBack((x,y))
    dangerous = newX>=908 and newX<=980 and newY>=160
    logger().debug("checking whether {}->{} is in forbidden zone, result: {}".format((x,y), (newX,newY), dangerous))
    return dangerous


def angle((x1,y1), (x2,y2)):
    p1 = (x1,y1)
    p2 = (x2,y2)
    if (x1==0 and y1==0) or (x2==0 and y2==0):
        cosine = 1
    else:
        cosine = np.dot(p1,p2) / np.linalg.norm(p1) / np.linalg.norm(p2)
    logger().debug("cosine between {} and {}: {}".format(p1,p2,cosine))
    return cosine


def trajshow(image, mapimage, trajs):
    """
    trajs       : [ (color, [ [(x,y,dx,dy,t)] ]) ]
    """
    cv2.putText(image, str(len(trajs)), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    for (color, objectTrajs) in trajs:  # objectTrajs: different trajectories of the same object
        last = (-100, -100)  # last trajectory's ending point

        logger().debug("trajectory has {} part(s)".format(len(objectTrajs)))

        for traj in objectTrajs:
            if distance(last, traj[0][0:2]) < 50:
                # come in again
                cv2.putText(mapimage, "anomaly: lost", (150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

            last = traj[-1][0:2]

            if forbidden(last):
                cv2.putText(mapimage, "anomaly: entered forbidden zone", (150,370), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

            # trajectory
            traj = [(x,y) for (x,y,dx,dy,t) in traj]
            # down sampling
            traj = list(downsample(traj, 10))
            # smooth
            traj = signal.medfilt2d(traj, kernel_size=(11,1))
    
            pairs = zip(traj, traj[1:])  # pairs: [((x1,y1),(x2,y2))]
            prevV = (0,0)
            suspicious = 0
            for ((x1,y1), (x2,y2)) in pairs:
                if angle(prevV, (x2-x1, y2-y1)) < cos(3.5*pi/7):
                    cv2.putText(mapimage, "anomaly: lost", (150,400), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
                prevV = (x2-x1, y2-y1)

#                if np.linalg.norm(prevV) < 2:
#                    suspicious += 1
#                    if suspicious > 20:
#                        cv2.putText(mapimage, "anomaly: suspicious", (150,340), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
#                else:
#                    suspicious = 0

                (x1,x2,y1,y2) = map(int, (x1,x2,y1,y2))  # float -> int
                cv2.line(image, (x1,y1), (x2,y2), color, 4)
                cv2.line(mapimage, projectBack((x1,y1)), projectBack((x2,y2)), color, 1)

def downsample(xs, rate):
    xs = iter(xs)
    yield next(xs)
    while True:
        try:
            for i in range(rate):
                x = next(xs)
            yield x
        except StopIteration:
            break


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


def onEdge(image, (x,y,w,h)):
    (height, width, _) = image.shape
    eps = 15
    return (x<=eps) or (y<=eps) or (x+w>=width-eps) or (y+h>=height-eps)


matrices = {
        "seq_1.mp4": (np.array([ [-0.0373,-0.2517]
                               , [0.0740,-0.0063]
                               ]),
                      np.array([ [341.3177], [117.4532] ])),
        "seq_2.mp4": (),
        "seq_3.mp4": (np.array([ [-0.1571,-0.7888]
                               , [0.0837,-0.0045]
                               ]),
                      np.array([ [1103.9], [90.4] ])),
        "seq_4.mp4": (np.array([ [0.1032,0.4599]
                               , [-0.0694,0.0167]
                               ]),
                      np.array([ [498.4242], [149.1891] ])),
        "seq_5.mp4": (np.array([ [0.1615,0.6650]
                               , [-0.1138,-0.0749]
                               ]),
                      np.array([ [446.2605], [248.8616] ])),
        "seq_6.mp4": (np.array([ [0.1359,0.0463]
                               , [-0.0112,0.4863]
                               ]),
                      np.array([ [800.5693], [-62.3029] ]))
}
matrix = None


projectionMat = {
        "seq_1.mp4": np.array([ [  0.0801,  -0.5200, -93.1937 ]
                              , [ -0.1391,  -0.6606, 147.5483 ]
                              , [  0.0001,  -0.0053,   1.0000 ]
                              ]) ,

        "seq_2.mp4": np.array([ [  0.0706,  -2.0966, 563.1875 ]
                              , [  0.1507,  -0.5757,  34.5255 ]
                              , [  0.0002,  -0.0046,   1.0000 ]
                              ]) ,

        "seq_3.mp4": np.array([ [ -0.1376,  -1.2041,   4264.4 ]
                              , [  0.9145,   1.2953,-123.5846 ]
                              , [  0.0001,   0.0133,   1.0000 ]
                              ]) ,

        "seq_4.mp4": np.array([ [  0.0565,  -5.3918,   1080.9 ]
                              , [  0.2242,  -0.9058,  95.9198 ]
                              , [  0.0001,  -0.0061,   1.0000 ]
                              ]) ,

        "seq_5.mp4": np.array([ [  0.9453,  -7.8280,   1164.7 ]
                              , [  0.3612,  -0.9651, -86.7122 ]
                              , [  0.0010,  -0.0080,   1.0000 ]
                              ]) ,

        "seq_6.mp4": np.array([ [ -0.0446,  -4.3285,   1083.2 ]
                              , [  0.0745,  -1.2872, 372.7987 ]
                              , [  0.0001,  -0.0045,   1.0000 ]
                              ]) ,
}


def projectBack((x,y)):
    name = os.path.basename(sys.argv[1])
    A = projectionMat[name]
    [[x], [y], [w]] = np.dot(A, np.array([[x], [y], [1.0]]))
    return (int(x/w), int(y/w))

def drawTrajectory(pklFilename, videoFilename):
    mapimage = cv2.imread("/home/zepp/Downloads/map.png")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    trajVideoFilename = os.path.join(os.path.dirname(pklFilename), os.path.basename(videoFilename) + "_traj.mp4")
    mapVideoFilename = os.path.join(os.path.dirname(pklFilename), os.path.basename(videoFilename) + "_map.mp4")
    logger().info("writing video to {} and {}".format(trajVideoFilename, mapVideoFilename))
    trajVideo = cv2.VideoWriter(trajVideoFilename, fourcc, 30.0, (1280,720))
    mapVideo = cv2.VideoWriter(mapVideoFilename, fourcc, 30.0, (1135,405))

    data = dataGenerator(videoFilename, pklFilename)
    trajs = None

    global index
    global matrix

    matrix = matrices[os.path.basename(videoFilename)]

    # index     : int
    # frame     : image
    # bboxes    : [ (x,y,w,h) ]
    for (index, frame, instances) in data:
        if not trajs:  # init motion models
            trajs = Trajectories(frame, [i for i in instances if not onEdge(frame, i.bbox)], KalmanFilterModel, index)
        else:
            # remove bboxes laying on the edge
            trajs.updateAll(frame, [i for i in instances if not onEdge(frame, i.bbox)], index)

        mapimg = mapimage.copy()

        # draw trajectories on this frame, as well as the map
        trajshow(frame, mapimg, trajs.extractAll())
        # draw bbox provided by upstream
        bboxesshow(frame, [i.bbox for i in instances])
        # show index on top-left corner
        cv2.putText(frame, str(index), (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        #cv2.imshow("trajectories", frame)
        #cv2.imshow("map", mapimg)
        #cv2.waitKey(int(dt*100))
        trajVideo.write(frame)
        mapVideo.write(mapimg)
        cv2.waitKey(1)

    trajVideo.release()
    mapVideo.release()


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
