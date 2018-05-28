#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  19  2018

@author: jingyi
"""

import cv2 as cv
import math
import numpy as np
import os

from utils import gaussian_pyramid


class GrayLevelModels(object):
    """
    A class to build gray level models for each point
    """

    def __init__(self, images, shapes, k=0):
        self.images = images
        self.shapes = shapes
        self.k = k

        if os.path.exists("./data/asm_gl_mean.npy"):
            self.glmean = np.load("./data/asm_gl_mean.npy")
        if os.path.exists("./data/asm_gl_cov.npy"):
            self.glcov = np.load("./data/asm_gl_cov.npy")
        else:
            self.glmean, self.glcov = self.glm(self.images, self.shapes)

    @staticmethod
    def reset_shapes(shapes, scalar):
        """ Resetting shapes coordination divided by scalar """
        for shape in shapes:
            shape /= scalar
        return shapes

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

        # create positives and negatives for k points on normal to current point
        i = 1
        while len(positives) < self.k:
            # the int coordination of the sample
            # ( images has no float pixel coordination )
            new = (int(current[0] - i * norm[0]), int(current[1] - i * norm[1]))
            if (new not in positives) and (new not in negatives):
                positives.append(new)
            i += 1

        i = 1
        while len(negatives) < self.k:
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

    def get_image_profiles(self, images, shapes):
        """ Creating 2*k profiles for all points in all images
        :return profiles size: num_pts × num_imgs × 2k_sampling
        """
        profiles = np.zeros((shapes[0].num_pts, len(images), 2 * self.k))
        # iterate over all images
        for i in range(len(images)):
            # width, height = y, x in an image
            image = images[i].T
            # get ith shape
            shape = shapes[i]
            # iterate over all points in ith shape
            for idx in range(len(shape.pts)):
                # for each point, get the profile from each image
                profiles[idx, i, :] = \
                    self.profile(image, (shape.pts[idx - 2], shape.pts[idx - 1], shape.pts[idx]))
            print("Already finished", i + 1, "images.")

        return profiles

    def glm(self, images, shapes):
        """ Creating gray level model for every image """
        # get all profiles of all points from all images
        profiles = self.get_image_profiles(images, shapes)
        # get the number of points
        num_pts = profiles.shape[0]

        glmean = np.zeros((num_pts, 2 * self.k))
        glcov = np.zeros((num_pts, 2 * self.k, 2 * self.k))
        # for all points, calculate the mean profiles over all images
        for idx in range(num_pts):
            # get profiles of all images of ith point and get the mean
            profile = profiles[idx]
            mean_profile = profile.mean(0)
            glmean[idx] = mean_profile
            # get the covariance matrix of all profiles
            # a row as an example
            bias = profile - mean_profile
            glcov[idx] = np.cov(bias, rowvar=False)

        np.save("./data/asm_gl_mean.npy", glmean)
        np.save("./data/asm_gl_cov.npy", glcov)
        return glmean, glcov

        '''def __glm_pyramid(self):
        """ Creating gray level model for every image pyramid """
        # create gaussian pyramid for every images
        pyramid = np.array([gaussian_pyramid(self.images[i], self.level)
                            for i in range(len(self.images))])
        # build gray level model
        glmodels = []
        for l in range(self.level):
            # lth level of pyramid of all images
            images = pyramid[:, l]
            # gaussian pyramid is created by half the width and height of the image per level
            # so the width of kth level of pyramid is WIDTH/2**l, the same as height
            shapes = self.__reset_shapes(self.shapes, 2 ** l)
            # build gray level model for all images all shapes
            glmodels.append(self.__glm(images, shapes))

        return glmodels'''