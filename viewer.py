#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  16  2018

@author: jingyi
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from structure import Shape

PLOT_ON = True


def show_points(image, shape):
    for pt in shape.pts:
        cv.circle(image, (int(pt.x), int(pt.y)), 2, (0, 255, 0), -1)


def plot(choice, *args):
    """ Plotting different stages of the project """
    if not PLOT_ON:
        return
    else:
        if choice == 'gpa':
            plot_gpa(*args)


def plot_gpa(mean, aligned_shapes):
    """ Plotting the gpa images """
    # plot mean shape in red
    mean = Shape.get_matrix(mean)
    mean_x = mean[:, 0]
    mean_y = mean[:, 1]
    plt.plot(mean_y, -mean_x, color='r', marker='o')
    # plot first 50 aligned shapes in scatter
    num = 50
    aligned = aligned_shapes[:num]
    for shape in aligned:
        for pt in shape.pts:
            plt.scatter(pt.y, -pt.x)
    axes = plt.gca()
    plt.show()