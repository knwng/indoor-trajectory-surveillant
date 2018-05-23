#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function

import sys
import cv2
import re
import numpy as np

import logging
import itertools

from filterpy.kalman import KalmanFilter

# frames per detection
fpd = 4
dt = 1/25 * fpd
index = 0  # just a global varible used in logging
#startingFrame = 460

logging.basicConfig(level=logging.DEBUG)

def logger():
    return logging.getLogger("(index={})".format(index))

colors = itertools.cycle([(255,0,0), (0,255,0), (0,0,255)])


def detectedBBoxes(filename):
    with open(filename,'r') as f:

        currLines = []
        currIdx   = startingFrame
        for line in f:
            pat = re.compile(r"(\d+) \| (\d+) (\d+) (\d+) (\d+)")
            (idx, x, xmax, y, ymax) = map(int, pat.match(line).groups())
            w = xmax - x
            h = ymax - y

            if idx == currIdx:
                currLines.append((x,y,w,h))
            else:  # idx != currIdx
                if currLines:
                    yield (currIdx, currLines)
                    currLines = []

                skippedCount = (idx-currIdx-1) // fpd
                for i in range(skippedCount):
                    yield(currIdx + (i+1)*fpd, [])

                currIdx = idx
                currLines = [(x,y,w,h)]

        if currLines:
            yield (currIdx, currLines)


class MotionModel(object):
    id = 0

    def __init__(self, frame, bbox):
        """
        frame       : image
        bbox        : (x,y,w,h)
        """
        self.id = MotionModel.id
        self.id += 1
        MotionModel.id += 1

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
    just use currently detected bbox
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


class Video(object):
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            raise ValueError("file {} does not exist".format(filename))
        # current position
        self.ptr = 0

    def seek(self, frameIdx):
        if type(frameIdx) != int or frameIdx <= self.ptr:
            raise ValueError("invalid frame index: {}".format(frameIdx))

        while self.ptr != frameIdx:
            (ok, frame) = self.cap.read()
            self.ptr += 1
            if not ok:
                return (False, None)

        return (ok, frame)


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

def similarityInNorm((x1,y1), (x2,y2)):
    """
    Euclidean norm, [0, +infty], smaller is better
    """
    return np.linalg.norm([x1-x2, y1-y2])


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
    video = Video(videoFilename)
    allBBoxes = detectedBBoxes(bboxesFilename)
    trajs = None

    global index

    for i in itertools.count():
        index = (i+1) * fpd + startingFrame

        (ok, frame) = video.seek(index)
        if not ok:
            logger().info("{} reaches eof".format(videoFilename))
            break

        try:
            (idx, bboxes) = allBBoxes.next()
            if idx != index:
                logger().error("idx ({}) != index ({})".format(idx, index))
        except StopIteration:
            bboxes = []

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
    startingFrame = int(sys.argv[3])
    drawTrajectory(bboxesFilename, videoFilename)
    #drawTrajectory("single_person_bboxes.txt", "single_person.avi")
    cv2.destroyAllWindows()
    #for (idx, bboxes) in detectedBBoxes("single_person_bboxes.txt"):
    #    print(idx)
    #    for (x,y,w,h) in bboxes:
    #        print("  ({}, {}, {}, {})".format(x, y, w, h))
