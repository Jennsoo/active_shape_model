#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7  2018

@author: jingyi

Including helpers to perform Generalized Orthogonal Procrustes Analysis.
"""

import math
import numpy as np
from copy import deepcopy

from structure import Shape


def get_mean_shape(shapes):
    """ Getting the mean shape of all """
    mean_s = shapes[0]
    for shape in shapes[1:]:
        mean_s = mean_s + shape
    return mean_s / len(shapes)


def translate_to_origin(shape):
    """ The translation step essentially moves all the shapes to a common center.
    The origin (0,0) is the most likely candidate to become that common center,
    yet not exclusively so. """
    s = shape
    center_x, center_y = Shape.get_centroid(s)
    for pt in s.pts:
        pt.x -= center_x
        pt.y -= center_y
    return s


def normalize(shape):
    """ Normalizing every shape to limit them in the same size """
    s = shape
    matrix = Shape.get_matrix(s.get_vector())
    norm_x = math.sqrt(sum(matrix[:, 0] ** 2))
    norm_y = math.sqrt(sum(matrix[:, 1] ** 2))
    for pt in s.pts:
        pt.x /= norm_x
        pt.y /= norm_y
    return s


def rotate_to_target(shape, target):
    """ Rotating the shape to align to the target with matrix Q.
    Singular value decomposition:
            USV' = T'X
               Q = VU'

    :parameter shape: A shape of shape class
    :parameter target: A vector of mean shape
    :return rotated shape vector
    """
    target_matrix = Shape.get_matrix(target)
    shape_matrix = Shape.get_matrix(shape.get_vector())
    u, s, vt = np.linalg.svd(target_matrix.T.dot(shape_matrix))
    vu = vt.T * u.T
    new_shape = shape_matrix.dot(vu)
    return Shape.turn_back_to_point(new_shape.flatten())


def align(shape, target):
    """ Aligns shapes according to a mean shape. Contains method
    to set a mean shape. Main method is the align method that takes a shape
    as input and aligns it to the mean shape according to the
    following steps:

    1. Translating the shape such that its centroid is situated in
    the origin (0,0)

    2. Scaling and Normalizing the image

    3. Rotating the image in order to align it with the mean

    :parameter shape: A shape of shape class
    :parameter target: A vector of mean shape
    :return aligned shape
    """
    translated = translate_to_origin(shape)
    scaled = normalize(translated)
    aligned = rotate_to_target(scaled, target)
    return aligned


def gpa(shapes, accuracy,target_shape=None):
    """ Performing GPA method until converged.

    1. Select one shape to be the approximate mean shape
    (i.e. the first shape in the set).

    2. Align the shapes to the approximate mean shape.
        . Calculate the centroid of each shape (or set of landmarks).
        . Align all shapes centroid to the origin.
        . Normalize each shapes centroid size.
        . Rotate each shape to align with the newest approximate mean.

    3. Calculate the new approximate mean from the aligned shapes.

    4. If the approximate mean from steps 2 and 3 are different
    then return to step 2, otherwise you have found the true mean shape of the set.

    details in paper: Procrustes Analysis
    """
    s = deepcopy(shapes)
    # if target shape vector is given, then align to it
    # otherwise, get first one in the training set as the seed,
    # and turn it to vector
    if target_shape is not None:
        mean = target_shape
    else:
        mean = s[0].get_vector()
    # initialize the mean vector to be zero
    new_mean = np.zeros_like(mean)
    # calculate the difference between mean and new mean vector
    mean_bias = sum(mean - new_mean)
    while mean_bias > accuracy:
        new_mean = mean
        for i in range(len(s)):
            s[i] = align(s[i], new_mean)
        mean = get_mean_shape(s).get_vector()
        mean_bias = sum(mean - new_mean)

    return s, mean