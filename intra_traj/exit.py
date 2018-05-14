#!/usr/bin/env python2

import numpy as np

def polygonArea(xs, ys):
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(xs, np.roll(ys,1)) - np.dot(ys, np.roll(xs,1)))

class Region(object):
    """
    a quadrangle region
    """
    def __init__(self, points):
        """
        points: [(x, y)] (len==4)
        """
        self.vertices = points

    def contains(self, x, y):
        vxs = [v[0] for v in self.vertices]
        vys = [v[1] for v in self.vertices]

        S  = polygonArea(vxs, vys)
        S1 = polygonArea(vxs[0:2]+[x], vys[0:2]+[y])
        S2 = polygonArea(vxs[1:3]+[x], vys[1:3]+[y])
        S3 = polygonArea(vxs[2:4]+[x], vys[2:4]+[y])
        S4 = polygonArea([vxs[3],vxs[0],x], [vys[3],vys[0],y])

        # S_{ABCD} = S_{PAB} + S_{PBC} + S_{PCD} + S_{PDA}
        return S == S1 + S2 + S3 + S4

class Exit(object):
    """
    represent an exit in the video
    """
    def __init__(self, region, nextID):
        pass
