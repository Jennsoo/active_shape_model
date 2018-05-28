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


IMAGE_DIR = "data//02_Outdoor"


def main():
    print("*** Loading images and shapes from data set...")
    shapes = PointsReader.read_points_dictionary(IMAGE_DIR)

    images = ImagesReader.read_images_dictionary(IMAGE_DIR, "png", gray=True)

    asm = ActiveShapeModel(images, shapes)

    np.save("./data/asm_mean_shape.npy", asm.mean)


if __name__ == '__main__':
    main()