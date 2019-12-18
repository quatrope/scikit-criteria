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

__doc__ = """Simplests method of multi-criteria"""

__all__ = [
    "WeightedSum",
    "WeightedProduct"]


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..validate import criteriarr
from ..base import Data
from .. import norm, rank
from ..utils.doc_inherit import doc_inherit

from ._dmaker import DecisionMaker


# =============================================================================
# FUNCTIONS
# =============================================================================

def wsum(nmtx, ncriteria, nweights):
    # invert the minimization criteria
    nmtx = norm.invert_min(nmtx, ncriteria, axis=0)

    # calculate raning by inner prodcut
    rank_mtx = np.inner(nmtx, nweights)
    points = np.squeeze(np.asarray(rank_mtx))

    return rank.rankdata(points, reverse=True), points


def wprod(nmtx, ncriteria, nweights):
    # invert the minimization criteria
    nmtx = norm.invert_min(nmtx, ncriteria, axis=0)

    # instead of multiply we sum the logarithms
    lmtx = np.log10(nmtx)

    # add the weights to the mtx
    rank_mtx = np.multiply(lmtx, nweights)

    points = np.sum(rank_mtx, axis=1)

    return rank.rankdata(points, reverse=True), points


# =============================================================================
# OO
# =============================================================================

class WeightedSum(DecisionMaker):
    r"""The weighted sum model (WSM) is the best known and simplest
    multi-criteria decision analysis for evaluating a number of alternatives
    in terms of a number of decision criteria. It is very important to state
    here that it is applicable only when all the data are expressed in exactly
    the same unit. If this is not the case, then the final result is equivalent
    to "adding apples and oranges." To avoid this problem a previous
    normalization step is necesary.

    In general, suppose that a given MCDA problem is defined on :math:`m`
    alternatives and :math:`n` decision criteria. Furthermore, let us assume
    that all the criteria are benefit criteria, that is, the higher the values
    are, the better it is. Next suppose that :math:`w_j` denotes the relative
    weight of importance of the criterion :math:`C_j` and :math:`a_{ij}` is
    the performance value of alternative :math:`A_i` when it is evaluated in
    terms of criterion :math:`C_j`. Then, the total (i.e., when all the
    criteria are considered simultaneously) importance of alternative
    :math:`A_i`, denoted as :math:`A_{i}^{WSM-score}`, is defined as follows:

    .. math::

        A_{i}^{WSM-score} = \sum_{j=1}^{n} w_j a_{ij},\ for\ i = 1,2,3,...,m

    For the maximization case, the best alternative is the one that yields
    the maximum total performance value.

    Notes
    -----

    If some criteria is for minimization, this implementation calculates the
    inverse.

    Parameters
    ----------

    mnorm : string, callable, optional (default="sum")
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

          - **e_.points**: Array where the i-nth element represent the
            importance of the i-nth alternative.

    References
    ----------

    .. [1] Fishburn, P. C. (1967). Letter to the editor—additive utilities
       with incomplete product sets: application to priorities and assignments.
       Operations Research, 15(3), 537-542.
    .. [2] Weighted sum model. In Wikipedia, The Free Encyclopedia. Retrieved
       from https://en.wikipedia.org/wiki/Weighted_sum_model
    .. [3] Tzeng, G. H., & Huang, J. J. (2011). Multiple attribute decision
       making: methods and applications. CRC press.

    """

    def __init__(self, mnorm="sum", wnorm="sum"):
        super(WeightedSum, self).__init__(mnorm=mnorm, wnorm=wnorm)

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = wsum(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


class WeightedProduct(DecisionMaker):
    r"""The weighted product model (WPM) is a popular multi-criteria decision
    analysis method. It is similar to the weighted sum model.
    The main difference is that instead of addition in the main mathematical
    operation now there is multiplication.

    In general, suppose that a given MCDA problem is defined on :math:`m`
    alternatives and :math:`n` decision criteria. Furthermore, let us assume
    that all the criteria are benefit criteria, that is, the higher the values
    are, the better it is. Next suppose that :math:`w_j` denotes the relative
    weight of importance of the criterion :math:`C_j` and :math:`a_{ij}` is
    the performance value of alternative :math:`A_i` when it is evaluated in
    terms of criterion :math:`C_j`. Then, the total (i.e., when all the
    criteria are considered simultaneously) importance of alternative
    :math:`A_i`, denoted as :math:`A_{i}^{WPM-score}`, is defined as follows:

    .. math::

        A_{i}^{WPM-score} = \prod_{j=1}^{n} a_{ij}^{w_j},\ for\ i = 1,2,3,...,m

    To avoid underflow, instead the multiplication of the values we add the
    logarithms of the values; so :math:`A_{i}^{WPM-score}`, is finally defined
    as:

    .. math::

        A_{i}^{WPM-score} = \sum_{j=1}^{n} w_j \log(a_{ij}),\
                            for\ i = 1,2,3,...,m

    For the maximization case, the best alternative is the one that yields
    the maximum total performance value.

    Notes
    -----

    The implementation works as follow:

    - If we have some values of any criteria < 0 in the alternative-matrix
      we add the minimimun value of this criteria to all the criteria.
    - If we have some 0 in some criteria all the criteria is incremented by 1.
    - If some criteria is for minimization, this implementation calculates the
      inverse.
    - Instead the multiplication of the values we add the
      logarithms of the values to avoid underflow.


    Parameters
    ----------

    mnorm : string, callable, optional (default="sum")
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

          - **e_.points**: Array where the i-nth element represent the
            importance of the i-nth alternative.


    References
    ----------

    .. [1] Bridgman, P.W. (1922). Dimensional Analysis. New Haven, CT, U.S.A.:
           Yale University Press.

    .. [2] Miller, D.W.; M.K. Starr (1969). Executive Decisions and Operations
           Research. Englewood Cliffs, NJ, U.S.A.: Prentice-Hall, Inc.

    .. [3] Wen, Y. (2007, September 16). Using log-transform to avoid underflow
           problem in computing posterior probabilities.
           from http://web.mit.edu/wenyang/www/log_transform_for_underflow.pdf

    """

    def __init__(self, mnorm="sum", wnorm="sum"):
        super(WeightedProduct, self).__init__(mnorm=mnorm, wnorm=wnorm)

    @doc_inherit
    def preprocess(self, data):
        non_negative = norm.push_negatives(data.mtx, axis=0)
        non_zero = norm.add1to0(non_negative, axis=0)
        nmtx = self._mnorm(non_zero, axis=0)
        ncriteria = criteriarr(data.criteria)
        nweights = (
            self._wnorm(data.weights, criteria=data.criteria)
            if data.weights is not None else
            np.ones(data.criteria.shape))
        return Data(mtx=nmtx, criteria=ncriteria, weights=nweights,
                    anames=data.anames, cnames=data.cnames)

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = wprod(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
