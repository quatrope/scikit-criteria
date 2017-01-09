#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """Most basic method of multi-criteria"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import norm, util, rank


# =============================================================================
# FUNCTIONS
# =============================================================================

def mdwsum(mtx, criteria, weights=None):

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
