#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  23  2018

@author: jingyi
"""

import cv2 as cv
import math
import numpy as np
import sys

from structure import Point
from structure import Shape
from viewer import plot
from pdm import PointDistributionModel
from glm import GrayLevelModels
from viewer import show_points


def sprint(shape):
    for p in shape.pts:
        print(p.x, p.y)


class ActiveShapeModel(object):
    """
    A class to construct training model
    """
    def __init__(self, images, shapes):
        self.images = images

        # make sure the shapes are valid
        print("*** Checking Shapes and Images...")
        self.__check_shapes(shapes)

        """"# build weight matrix for every point in shapes
        print("Building weight matrix...")
        self.w = self.__get_weight_matrix()"""
        print("*** Setting up Active Shape Model...")

        # 1. train point distribution model
        print("--- Training Point Distribution Model...")
        pdmodel = PointDistributionModel(shapes)
        self.mean = pdmodel.mean
        print("------ Plotting GPA Model...")
        #plot('gpa', pdmodel.mean, pdmodel.aligned)

        # 2. train gray level model
        print("--- Training Gray Level Model pyramid...")
        glmodels = GrayLevelModels(self.images, shapes, 3)

        print("Done.")

    @staticmethod
    def __check_shapes(shapes):
        """ Checking if shape is valid (number of points is correct) """
        num_pts = shapes[0].num_pts
        for shape in shapes:
            if shape.num_pts is not num_pts:
                raise Exception("The shape has incorrect number of points.")

    def __get_weight_matrix(self):
        """ A weight matrix is defined as follows:
        let Rkl be the distance between points k and l in a shape;
        let VRkl be the variance in this distance over the set of
        shapes:

        wk = (sum(VRkl for l in range(1, n)))^-1

        details in Paper: Active Shape Models, p42

        :return: The weight matrix of shapes
        """

        # return empty matrix if no shapes in
        if not self.shapes:
            return np.array()

        num_pts = self.shapes[0].num_pts

        # calculate distances between one point to each other point
        # each shape has a size of (num_pts*num_pts) distance matrix
        distances = np.zeros((len(self.shapes), num_pts, num_pts))
        for s, shape in enumerate(self.shapes):
            for k in range(num_pts):
                for l in range(num_pts):
                    distances[s, k, l] = shape.pts[k].distance(shape.pts[l])

        # calculate weight according to distances
        w = np.zeros(num_pts)
        for k in range(num_pts):
            for l in range(num_pts):
                # add points' variance in the same order within all shapes
                w[k] += np.var(distances[:, k, l])

        return 1/w
