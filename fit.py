#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  17  2018

@author: jingyi
"""

import cv2 as cv
import math
import numpy as np
import os
import sys

from structure import Point
from structure import Shape
from viewer import show_points


class ModelFitter(object):
    """
    A class to fit the model to images
    """
    def __init__(self, image, shape, n=0, k=0):
        self.image = self.__cvtGradient(image)
        self.shape = shape
        self.k = k  # length of a feature
        self.n = n  # whole features

        self.__load_features()

    @staticmethod
    def __cvtGradient(image):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = np.sqrt(image.astype(float))
        dx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
        dy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
        absx = cv.convertScaleAbs(dx)
        absy = cv.convertScaleAbs(dy)
        # gradient image
        gimage = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
        return gimage

    def __load_features(self):
        """ Loading trained features from asm """
        if os.path.exists("./data/asm_gl_mean.npy"):
            self.glmean = np.load("./data/asm_gl_mean.npy")
        if os.path.exists("./data/asm_gl_cov.npy"):
            self.glcov = np.load("./data/asm_gl_cov.npy")
        if os.path.exists("./data/P.npy"):
            self.P = np.load("./data/P.npy")
            #self.P = self.P.astype(float)
        else:
            print("No trained file.")

    @staticmethod
    def get_normal_to_point(pre, nex):
        """ Getting normal of current with previous and next points """
        dx = nex.x - pre.x
        dy = nex.y - pre.y
        # turn the normal vector to a unit
        mag = math.sqrt(dx ** 2 + dy ** 2)
        mag = mag if mag > 0 else 1
        return -dy / mag, dx / mag

    def sample_along_normal(self, points):
        """ Sampling along the normal to points[1]
        :return the profile list of points[1]
        """
        # get normal of point[1] with point[0] and point[2]
        norm = self.get_normal_to_point(points[0], points[2])
        if abs(norm[0]) == 0 and abs(norm[1]) == 0:
            norm = (0, 1)
        positives = []
        negatives = []
        # current point is points[1]
        current = (int(points[1].x), int(points[1].y))

        # create positives and negatives for n points on normal to current point
        i = 1
        while len(positives) < self.n:
            # the int coordination of the sample
            # ( images has no float pixel coordination )
            new = (int(current[0] - i * norm[0]), int(current[1] - i * norm[1]))
            if (new not in positives) and (new not in negatives):
                positives.append(new)
            i += 1

        i = 1
        while len(negatives) < self.n:
            new = (int(current[0] + i * norm[0]), int(current[1] + i * norm[1]))
            if (new not in positives) and (new not in negatives):
                negatives.append(new)
            i += 1

        # merge them in order to be a (2k+1) length profile list
        negatives.reverse()
        return negatives + [current] + positives

    def profile(self, image, points):
        """ Get the profile for points[1] in an image """
        # gray features of points[1], len = 2k+1
        height, width = image.shape[:2]
        grays = []
        for x, y in self.sample_along_normal(points):
            if x >= height or x < 0 or y >= width or y < 0:
                grays.append(0)
            else:
                grays.append(float(image[x, y]))
        grays = np.array(grays)
        # get difference of gray features element by element, so len = 2k
        grays_diff = np.diff(grays)
        # normalize
        div = sum(abs(grays_diff)) if sum(abs(grays_diff)) > 0 else 1
        return grays_diff / div

    def glm(self):
        """ Creating gray level model for current image """
        profiles = np.zeros((self.shape.num_pts, 2 * self.n))
        # iterate over all points in ith shape
        for idx in range(self.shape.num_pts):
            # for each point, get the profile from each image
            profiles[idx, :] = \
                self.profile(self.image, (self.shape.pts[idx - 2], self.shape.pts[idx - 1], self.shape.pts[idx]))
        return profiles

    def find_closest(self):
        """ Finding closest feature """
        pf = self.glm()
        new_pf = []
        for i in range(pf.shape[0]):
            new_pf_tmp = []
            for j in range(2*(self.n-self.k)+1):
                new_pf_tmp.append(pf[i, j:j+2*self.k])
            new_pf.append(new_pf_tmp)
        new_pf = np.array(new_pf)

        offset = []
        # computing m distance
        for i in range(new_pf.shape[0]):
            features = []
            for j in range(new_pf.shape[1]):
                f_tmp = new_pf[i, j, :]

                S = np.zeros_like(self.glcov[i])
                try:
                    S = np.linalg.inv(self.glcov[i])
                except np.linalg.LinAlgError:
                    pass
                else:
                    S = np.ones_like(self.glcov[i])*sys.maxsize

                m = (f_tmp-self.glmean[i]).dot(S).dot((f_tmp-self.glmean[i]).T)
                features.append(m)
            min_idx = features.index(min(features))
            off_idx = int(new_pf.shape[1]/2-min_idx)
            if i+1 == new_pf.shape[0]:
                idx = -1
            else:
                idx = i
            # get normal of current point
            norm = self.get_normal_to_point(self.shape.pts[idx-1], self.shape.pts[idx+1])
            offset.append([norm[0]*off_idx, norm[1]*off_idx])
        return np.array(offset).flatten()

    def iterate(self, iter):
        """ Iterating to get updates """
        b = 0
        for i in range(iter):
            svec = self.shape.get_vector()
            # db = P.T * dx
            delta_b = self.P.T.dot(self.find_closest())
            b += delta_b
            # limit b
            b[b > math.sqrt(3)] = math.sqrt(3)
            b[b < -math.sqrt(3)] = -math.sqrt(3)
            delta = abs(self.P.dot(b))
            svec += delta.astype(float)
            self.shape = Shape.turn_back_to_point(svec)

    def update_shape(self):
        self.iterate(5)
