import os

import cv2

import numpy as np
from convert import walk_img, imread


if __name__ == '__main__':
    dir = "data/images"
    for name in walk_img(dir):
        img = imread(name)
        cv2.imshow('', img)
        code = cv2.waitKey(0)
        print(code, ord("a"), ord("d"))
