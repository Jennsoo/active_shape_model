#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  16  2018

@author: jingyi
"""

import math
import numpy as np


class Point(object):
    """
    A class to represent point
    """
    def __init__(self, x, y):
        """ Initialized by coordinate x and y """
        self.x = x
        self.y = y

    def __add__(self, other):
        """ Adding corresponding parameters of object Point """
        return Point(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        """ Subtracted corresponding parameters by other point """
        return Point(self.x-other.x, self.y-other.y)

    def __truediv__(self, other):
        """ Dividing each parameter of object Point by a constant """
        return Point(self.x/other, self.y/other)

    def __eq__(self, other):
        """ Checking if is equal to another """
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        """ Checking it is not equal to another """
        return not self.__eq__(other)

    def distance(self, other):
        """ Calculating distance to another point """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Shape(object):
    """
    A class to represent the shape of an object by a list of points
    """
    def __init__(self, pts=[]):
        """ Initialized by a list of points """
        self.pts = pts
        self.num_pts = len(pts)

    def __add__(self, other):
        """ Adding two shapes equals to adding corresponding points """
        s = Shape([])
        for i, p in enumerate(self.pts):
            s.append_shape_with_points(p+other.pts[i])
        return s

    def __sub__(self, other):
        """ Subtracting other shapes equals to subtracting corresponding points """
        s = Shape([])
        for i, p in enumerate(self.pts):
            s.append_shape_with_points(p-other.pts[i])
        return s

    def __truediv__(self, other):
        """ Dividing a shape equals to dividing every point by a constant"""
        s = Shape([])
        for p in self.pts:
            s.append_shape_with_points(p/other)
        return s

    def __eq__(self, other):
        """ Checking if is equal to another shape """
        for i in range(len(self.pts)):
            if self.pts[i] is not other.pts[i]:
                return False
        return True

    def __ne__(self, other):
        """ Checking if is not equal to another shape """
        return not self.__eq__(other)

    def slice(self, index):
        """ Returning a slice of a shape """
        s = Shape([])
        for i in range(index):
            s.append_shape_with_points(Point(self.pts[i].x, self.pts[i].y))
        return s

    def merge(self, another):
        """ Merging one shape with another"""
        s = self
        for pt in another.pts:
            self.append_shape_with_points(pt)
        return s

    def append_shape_with_points(self, pt):
        """ Appending points to a shape """
        self.pts.append(pt)
        self.num_pts += 1

    def transform(self, trans):
        """ Transforming the shape with Point(x, y) """
        s = Shape([])
        for p in self.pts:
            s.append_shape_with_points(p+trans)
        return s

    def get_centroid(self):
        """ Getting centroid of pts on x and y axis """
        x = sum([self.pts[k].x for k in range(len(self.pts))])
        y = sum([self.pts[k].y for k in range(len(self.pts))])
        return x/len(self.pts), y/len(self.pts)

    """ Different interpretation of a shape """
    def get_vector(self):
        """ Changing shapes to a vector: getting the list of points'
        coordinates, then flat it to one-dimensional vector """
        vec = np.zeros((self.num_pts, 2))
        for i in range(len(self.pts)):
            vec[i, :] = [self.pts[i].x, self.pts[i].y]
        return vec.flatten()

    @staticmethod
    def get_matrix(vec):
        """ Changing shapes from vector to n*2 matrix """
        return np.reshape(vec, (-1, 2))

    @staticmethod
    def turn_back_to_point(vec):
        """ Turning vector back to class Point matrix """
        s = Shape([])
        vec = np.reshape(vec, (-1, 2))
        for i, j in vec:
            s.append_shape_with_points(Point(i, j))
        return s

    def print(self):
        """ Printing the whole shape """
        for p in self.pts:
            print(p.x, p.y)
        print('\n')
