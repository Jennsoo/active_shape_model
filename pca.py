#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7  2018

@author: jingyi

Including helpers to perform Principal Component Analysis.
"""

import numpy as np


def pca(samples, accuracy):
    """ The principal axes of a 2n-D ellipsoid fitted to the
    data can be calculated by applying a principal component
    analysis (PCA) to the data.

    1. Zero-meaning: Each row as an example and each col as
    a feature, calculating the means of every col. Make them
    subtracted by every number in the corresponding cols.

    2. Getting covariance matrix: Using .cov function in numpy.
    Parameter rowvar = False means a row as an example. rowvar
    = True means a col as an example. We here use rowvar = False.

    3. Getting eigenvalues and feature-vectors. Using .eig
    function in numpy.linalg to directly calculate.

    4. Keeping the main part of data. Sorting the eigenvalues,
    calculating the formula:

        sum(lambda[:k])/sum(lambda)

    Getting k which makes it greater than one accuracy.

    :return: The first k eigenvalues with feature-vectors,
    the vector of mean shape, the first k main modes of variation.
    """
    # get the mean shape vector of all shapes
    mean_vector = np.mean(samples, axis=0)
    # turn the vector back to origin
    mean_vector = np.reshape(mean_vector, (-1, 2))
    # get the mean of every feature
    mean_x = mean_vector[:, 0].mean()
    mean_y = mean_vector[:, 1].mean()
    # subtracted by every number in the corresponding cols
    mean_vector[:, 0] = [x - mean_x for x in mean_vector[:, 0]]
    mean_vector[:, 1] = [y - mean_y for y in mean_vector[:, 1]]
    # flatten to 1D vector
    mean_vector = mean_vector.flatten()

    # get covariance matrix
    cov_mat = np.cov(samples, rowvar=False)

    # get eigenvalues and feature-vectors
    evals, evecs = np.linalg.eig(cov_mat)

    # get first t modes in all shapes
    t = 0
    for i in range(len(evals)):
        if sum(evals[:i]) / sum(evals) >= accuracy:
            break
        else:
            t += 1
    return evals[:t], evecs[:t], mean_vector