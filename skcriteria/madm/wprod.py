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

from .. import norm, util, rank


# =============================================================================
# FUNCTIONS
# =============================================================================

def wprod(mtx, criteria, weights=None):
    """The weighted product model (WPM) is a popular multi-criteria decision
    analysis (MCDA) / multi-criteria decision making (MCDM) method. It is
    similar to the weighted sum model (WSM). The main difference is that
    instead of addition in the main mathematical operation now there is
    multiplication.


    Notes
    -----

    The implementation works as follow:

    - If we have some values of any criteria < 0 in the alternative-matrix
      we add the minimimun value of this criteria to all the criteria.
    - If we have some 0 in some criteria all the criteria is incremented by 1.
    - Instead the multiplication of the values we add the
      logarithms of the values to avoid underflow.


    References
    ----------

    Bridgman, P.W. (1922). Dimensional Analysis. New Haven, CT, U.S.A.:
    Yale University Press.

    Miller, D.W.; M.K. Starr (1969). Executive Decisions and Operations
    Research. Englewood Cliffs, NJ, U.S.A.: Prentice-Hall, Inc.

    Wen, Y. (2007, September 16). Using log-transform to avoid underflow
    problem in computing posterior probabilities. Retrieved January 7, 2017,
    from http://web.mit.edu/wenyang/www/log_transform_for_underflow.pdf

    """

    # normalize
    ncriteria = util.criteriarr(criteria)
    nweights = norm.sum(weights) if weights is not None else 1

    # push all negative values to be > 0 by criteria
    non_negative = norm.push_negatives(mtx, axis=0)
    non_zero = norm.add1to0(non_negative, axis=0)
    nmtx = norm.sum(non_zero, axis=0)

    # invert the minimization criteria
    if util.MIN in ncriteria:
        mincrits = np.squeeze(np.where(ncriteria == util.MIN))
        mincrits_inverted = 1.0 / nmtx[:, mincrits]
        nmtx = nmtx.astype(mincrits_inverted.dtype.type)
        nmtx[:, mincrits] = mincrits_inverted

    # calculate raning by inner prodcut
    lmtx = np.log(nmtx)
    rank_mtx = np.multiply(lmtx, nweights)

    points = np.sum(rank_mtx, axis=1)

    return rank.rankdata(points, reverse=True), points


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
