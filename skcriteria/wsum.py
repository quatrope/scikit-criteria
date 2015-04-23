#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> wrote this file. As long as you retain this notice
# you can do whatever you want with this stuff. If we meet some day, and you
# think this stuff is worth it, you can buy me a WISKEY in return Juan BC


# =============================================================================
# DOCS
# =============================================================================

"""Most basic method of multi-criteria, probably you never want to use this
methodology.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .common import norm, util, rank


# =============================================================================
# FUNCTIONS
# =============================================================================

def wsum(mtx, criteria, weights=None):

    nmtx = norm.sum(mtx, axis=0)
    ncriteria = util.criteriarr(criteria)
    nweights = norm.sum(weights) if weights is not None else 1

    # add criteria to weights
    cweights = nweights * ncriteria

    # calculate raning by inner prodcut
    rank_mtx = np.inner(nmtx, cweights)
    points = np.squeeze(np.asarray(rank_mtx))

    return rank.rankdata(points, reverse=True), points


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
