#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from skcriteria.common import norm, util, rank


# =============================================================================
# TOPSIS
# =============================================================================

def topsis(mtx, criteria, weights=1):

    # This guarantee the criteria array consistency
    ncriteria = util.criteriarr(criteria)

    # normalize mtx
    nmtx = norm.vector(mtx, axis=0)

    # apply weights
    nweights = norm.vector(weights) if weights is not None else 1
    wmtx = np.multiply(nmtx, nweights)

    # extract mins and maxes
    mins = np.min(wmtx, axis=0)
    maxs = np.max(wmtx, axis=0)

    # create the ideal and the anti ideal arrays
    ideal = np.where(ncriteria == util.MAX, maxs, mins)
    anti_ideal = np.where(ncriteria == util.MIN, maxs, mins)

    # calculate distances
    d_better = np.sqrt(np.sum(np.power(wmtx - ideal, 2), axis=1))
    d_worst = np.sqrt(np.sum(np.power(wmtx - anti_ideal, 2), axis=1))

    # relative closeness
    closeness = d_worst / (d_better + d_worst)

    # compute the rank and return the result
    return rank.rankdata(closeness, reverse=True), closeness

