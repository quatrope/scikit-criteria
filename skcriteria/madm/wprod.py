#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


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
from ..dmaker import DecisionMaker


# =============================================================================
# FUNCTIONS
# =============================================================================

def wprod(nmtx, ncriteria, nweights):
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
# OO
# =============================================================================

class WeightedProduct(DecisionMaker):
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

    def __init__(self, mnorm="sum", wnorm="sum"):
        super(WeightedProduct, self).__init__(mnorm=mnorm, wnorm=wnorm)

    def normalize(self, mtx, criteria, weights):
        # push all negative values to be > 0 by criteria
        non_negative = norm.push_negatives(mtx, axis=0)
        non_zero = norm.add1to0(non_negative, axis=0)
        return super(WeightedProduct, self).normalize(
            non_zero, criteria, weights)

    def solve(self, *args, **kwargs):
        rank, points = wprod(*args, **kwargs)
        return None, rank, {"points": points}
