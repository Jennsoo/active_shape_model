#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  16  2018

@author: jingyi
"""

import glob
import os
import cv2 as cv

from structure import Point
from structure import Shape


class PointsReader(object):
    """
    A class to read points list from training example
    """
    @staticmethod
    def open_file(file_name):
        """ Reading points to form a shape and return it """
        s = Shape([])
        num_pts = 0
        # open the .pts file and read points
        with open(file_name) as file:
            first_line = file.readline()
            # handle useless info
            if first_line.startswith("version"):
                # read "n_points"
                num_pts = int(file.readline().split()[1])
                # drop "{""
                file.readline()

            # handle points coordinates info
            for line in file:
                # read until "}"
                if not line.startswith("}"):
                    # read a line, remove space and split x, y axes
                    # x_coordinate = float(pt[0])
                    # y_coordinate = float(pt[1])
                    pt = line.strip().split()
                    s.append_shape_with_points(Point(float(pt[0]), float(pt[1])))

        # export errors when file is unexpected
        if num_pts is not s.num_pts:
            print("Unexpected number of points in file.")

        return s

    @staticmethod
    def read_points_dictionary(file_path):
        """ Reading all the .pts file as a dictionary """
        pts_dic = []
        # find all the .pts in file_path
        for file in sorted(glob.glob(os.path.join(file_path, "*.pts"))):
            pts_dic.append(PointsReader.open_file(file))
        return pts_dic

    @staticmethod
    def merge_shape_lists(shapes, others):
        """ Appending shape list with another shape list """
        for shape in others:
            shapes.append(shape)
        return shapes


class ImagesReader(object):
    """
    A class to read image
    """
    @staticmethod
    def open_file(file_name, gray=False):
        """ Reading a gray image """
        img = cv.imread(file_name)
        if gray:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def read_images_dictionary(file_path, extension, gray=False):
        """ Reading image from file path and return all """
        imgs_dic = []
        for file in sorted(glob.glob(os.path.join(file_path, "*."+extension))):
            imgs_dic.append(ImagesReader.open_file(file, gray))
        return imgs_dic

    @staticmethod
    def merge_image_list(images, others):
        """ Appending image list with another image list"""
        for image in others:
            images.append(image)
        return images