#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8  2018

@author: jingyi
"""

import cv2 as cv

from loader import PointsReader
from loader import ImagesReader
from asm import ActiveShapeModel
from structure import Shape
import numpy as np
import math
import time
from fit import ModelFitter
import matplotlib.pyplot as plt
from viewer import show_points
from utils import get_init_shape


# asm pre
MEAN_SHAPE_PATH = "./data/asm_mean_shape.npy"


def main():
    mean_shape = Shape.turn_back_to_point(np.load(MEAN_SHAPE_PATH))
    for pt in mean_shape.pts:
        (pt.x, pt.y) = (pt.y, pt.x)

    img = cv.imread("./data/test.png")

    mean_shape = get_init_shape(img, mean_shape)

    print("*** Fitting asm to a test image...")
    fit = ModelFitter(img, mean_shape, n=15, k=3)
    fit.update_shape()
    print("Done.")

    show_points(img, fit.shape)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()