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
# DOCS
# =============================================================================

"""Methods based on an aggregating function representing
“closeness to the ideal”.


"""

__all__ = ['TOPSIS']


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import rank
from ..validate import MIN, MAX
from ..utils.doc_inherit import doc_inherit

from ._dmaker import DecisionMaker


# =============================================================================
# Function
# =============================================================================

def topsis(nmtx, ncriteria, nweights):

    # apply weights
    wmtx = np.multiply(nmtx, nweights)

    # extract mins and maxes
    mins = np.min(wmtx, axis=0)
    maxs = np.max(wmtx, axis=0)

    # create the ideal and the anti ideal arrays
    ideal = np.where(ncriteria == MAX, maxs, mins)
    anti_ideal = np.where(ncriteria == MIN, maxs, mins)

    # calculate distances
    d_better = np.sqrt(np.sum(np.power(wmtx - ideal, 2), axis=1))
    d_worst = np.sqrt(np.sum(np.power(wmtx - anti_ideal, 2), axis=1))

    # relative closeness
    closeness = d_worst / (d_better + d_worst)

    # compute the rank and return the result
    return rank.rankdata(closeness, reverse=True), ideal, anti_ideal, closeness


# =============================================================================
# OO
# =============================================================================

class TOPSIS(DecisionMaker):
    """TOPSIS is based on the concept that the chosen alternative should have
    the shortest geometric distance from the ideal solution
    and the longest euclidean distance from the worst solution.

    An assumption of TOPSIS is that the criteria are monotonically increasing
    or decreasing, and also allow trade-offs between criteria, where a poor
    result in one criterion can be negated by a good result in another
    criterion.

    Parameters
    ----------

    mnorm : string, callable, optional (default="vector")
        Normalization method for the alternative matrix.

    wnorm : string, callable, optional (default="sum")
        Normalization method for the weights array.

    Returns
    -------

    Decision : :py:class:`skcriteria.madm.Decision`
        With values:

        - **kernel_**: None
        - **rank_**: A ranking (start at 1) where the i-nth element represent
          the position of the i-nth alternative.
        - **best_alternative_**: The index of the best alternative.
        - **alpha_solution_**: True
        - **beta_solution_**: False
        - **gamma_solution_**: True
        - **e_**: Particular data created by this method.

          - **e_.closeness**: Array where the i-nth element represent the
            closenees of the i-nth alternative to ideal and worst solution.

    References
    ----------

    .. [1] Yoon, K., & Hwang, C. L. (1981). Multiple attribute decision
       making: methods and applications. SPRINGER-VERLAG BERLIN AN.
    .. [2] TOPSIS. In Wikipedia, The Free Encyclopedia. Retrieved
       from https://en.wikipedia.org/wiki/TOPSIS
    .. [3] Tzeng, G. H., & Huang, J. J. (2011). Multiple attribute decision
       making: methods and applications. CRC press.

    """

    def __init__(self, mnorm="vector", wnorm="sum"):
        super(TOPSIS, self).__init__(mnorm=mnorm, wnorm=wnorm)

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, ideal, anti_ideal, closeness = topsis(nmtx, ncriteria, nweights)
        extra = {
            "ideal": ideal,
            "anti_ideal": anti_ideal,
            "closeness": closeness}
        return None, rank, extra
