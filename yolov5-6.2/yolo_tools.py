from collections import OrderedDict
from typing import List, Union
import numpy as np


def calc_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p2) - np.array(p1)) ** 2))


class BBox:
    def __init__(self, box):
        self.bbox = box
        self.xmin, self.ymin, self.xmax, self.ymax = box

    @property
    def diagonal(self):
        return calc_dist(self.bbox[:2], self.bbox[2:4])

    @property
    def x1(self):
        return self.xmin

    @property
    def x2(self):
        return self.xmax

    @property
    def y1(self):
        return self.ymin

    @property
    def y2(self):
        return self.ymax

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    @property
    def cx(self):
        return (self.xmax + self.xmin) / 2

    @property
    def cy(self):
        return (self.ymin + self.ymax) / 2
