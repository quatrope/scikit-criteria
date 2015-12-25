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

__doc__ = ""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .common import norm, util, rank


# =============================================================================
# FUNCTIONS
# =============================================================================

def wprod(mtx, criteria, weights=None):
    """The weighted product model (WPM) is a popular multi-criteria decision
    analysis (MCDA) / multi-criteria decision making (MCDM) method. It is
    similar to the weighted sum model (WSM). The main difference is that
    instead of addition in the main mathematical operation now there is
    multiplication.

    References
    ----------

    Bridgman, P.W. (1922). Dimensional Analysis. New Haven, CT, U.S.A.:
    Yale University Press.

    Miller, D.W.; M.K. Starr (1969). Executive Decisions and Operations
    Research. Englewood Cliffs, NJ, U.S.A.: Prentice-Hall, Inc.

    """

    # normalize
    ncriteria = util.criteriarr(criteria)
    nweights = norm.sum(weights) if weights is not None else 1

    if util.MIN in ncriteria:
        mtx = np.asarray(mtx)
        mincrits = np.squeeze(np.where(ncriteria == util.MIN))
        mincrits_inverted = 1.0 / mtx[:, mincrits]
        mtx = mtx.astype(mincrits_inverted.dtype.type)
        mtx[:, mincrits] = mincrits_inverted

    nmtx = norm.sum(
        norm.push_negatives(norm.eps(mtx, axis=0), axis=0), axis=0
    )

    # calculate raning by inner prodcut
    rank_mtx = np.power(nmtx, nweights)
    points = np.prod(rank_mtx, axis=1)

    return rank.rankdata(points, reverse=True), points


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
