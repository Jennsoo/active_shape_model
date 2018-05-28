#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes Mar  21  2018

@author: jingyi
"""

import numpy as np
import cv2 as cv
import math

from structure import Shape


def gaussian_pyramid(image, level=0):
    """ Creating gray level pyramid with gaussian pyramid """
    # gaussian pyramid
    pyramid = [image]
    for l in range(level):
        pyramid.append(cv.pyrDown(pyramid[l]))
    return pyramid


def get_init_shape(img, mean_shape):
    """ Getting initial shape from eyes """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier("./data/haarcascade_eye.xml")

    center = []
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            center.append((x + ex + ew / 2, y + ey + eh / 2))
            # cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # cv.circle(roi_color, (int(ex + ew / 2), int(ey + eh / 2)), 2, (0, 255, 0), 2)

    center = sorted(center)
    true_left_center = center[0]
    true_right_center = center[1]

    left_shape = Shape.get_matrix(mean_shape.get_vector())[36:42]
    right_shape = Shape.get_matrix(mean_shape.get_vector())[42:48]

    left_shape = np.array(left_shape)
    right_shape = np.array(right_shape)

    left_center = np.mean(left_shape, axis=0)
    right_center = np.mean(right_shape, axis=0)

    # translation
    t = true_left_center - left_center

    # scale
    true_dist = math.sqrt(
        (true_right_center[1] - true_left_center[1]) ** 2 + (true_right_center[0] - true_left_center[0]) ** 2)
    dist = math.sqrt((right_center[1] - left_center[1]) ** 2 + (right_center[0] - left_center[0]) ** 2)
    s = true_dist / dist

    # theta
    k1 = (true_right_center[1] - true_left_center[1]) / (true_right_center[0] - true_left_center[0])
    k2 = (right_center[1] - left_center[1]) / (right_center[0] - left_center[0])
    theta = math.atan((k1 - k2) / (1 + k1 * k2))
    theta = math.radians(theta)

    for pt in mean_shape.pts:
        pt.x = pt.x * math.cos(theta) + pt.y * math.sin(theta)
        pt.y = -pt.x * math.sin(theta) + pt.y * math.cos(theta)
        pt.x = int(pt.x * s)
        pt.y = int(pt.y * s)

    left_shape = Shape.get_matrix(mean_shape.get_vector())[36:42]
    left_shape = np.array(left_shape)
    left_center = np.mean(left_shape, axis=0)
    t = true_left_center - left_center
    for pt in mean_shape.pts:
        pt.x = int(pt.x + t[0])
        pt.y = int(pt.y + t[1])

    return mean_shape