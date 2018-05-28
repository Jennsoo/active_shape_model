#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  19  2018

@author: jingyi
"""

import numpy as np
import math
from copy import deepcopy
import os

from structure import Shape
from pca import pca
from gpa import gpa


class PointDistributionModel(object):
    """
    A class to build point distribution model with pga """
    def __init__(self, shapes):
        # align every shape to the mean shape
        self.aligned, self.mean = gpa(shapes, 0.000001)
        # construct the asm model with pca
        self.evals, self.evecs, self.mean = pca(self.__get_shape_vectors(self.aligned), 0.99)
        np.save("./data/P.npy", self.evecs.T)

    @staticmethod
    def __get_shape_vectors(shapes):
        """ Getting a vector matrix from all shapes """
        shape_vectors = np.array([s.get_vector() for s in shapes])
        return shape_vectors