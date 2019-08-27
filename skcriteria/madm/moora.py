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

__doc__ = """Implementation of a family of Multi-objective optimization on
the basis of ratio analysis (MOORA) methods.

"""

__all__ = [
    "RatioMOORA",
    "RefPointMOORA",
    "FMFMOORA",
    "MultiMOORA"]


# =============================================================================
# IMPORTS
# =============================================================================

import itertools

import numpy as np

from ..validate import MIN, MAX, criteriarr
from ..base import Data
from .. import norm, rank

from ..utils.doc_inherit import doc_inherit

from ._dmaker import DecisionMaker


# =============================================================================
# FUNCTIONS
# =============================================================================

def ratio(nmtx, ncriteria, nweights):

    # invert the minimization criteria
    cweights = nweights * ncriteria

    # calculate raning by inner prodcut
    rank_mtx = np.inner(nmtx, cweights)
    points = np.squeeze(np.asarray(rank_mtx))
    return rank.rankdata(points, reverse=True), points


def refpoint(nmtx, criteria, weights):
    # max and min reference points
    rpmax = np.max(nmtx, axis=0)
    rpmin = np.min(nmtx, axis=0)

    # merge two reference points acoording criteria
    mask = np.where(criteria == MAX, criteria, 0)
    rpoints = np.where(mask, rpmax, rpmin)

    # create rank matrix
    rank_mtx = np.max(np.abs(weights * (nmtx - rpoints)), axis=1)
    points = np.squeeze(np.asarray(rank_mtx))
    return rank.rankdata(points), points


def fmf(nmtx, criteria, weights):
    lmtx = np.multiply(np.log(nmtx), weights)

    if np.all(np.unique(criteria) == [MAX]):
        # only max
        points = np.sum(lmtx, axis=1)
    elif np.all(np.unique(criteria) == [MIN]):
        # only min
        points = 1 - np.sum(lmtx, axis=1)
    else:
        # min max
        min_mask = np.ravel(np.argwhere(criteria == MAX))
        max_mask = np.ravel(np.argwhere(criteria == MIN))

        # remove invalid values
        min_arr = np.delete(lmtx, min_mask, axis=1)
        max_arr = np.delete(lmtx, max_mask, axis=1)

        mins = np.sum(min_arr, axis=1)
        maxs = np.sum(max_arr, axis=1)
        points = maxs - mins

    return rank.rankdata(points, reverse=True), points


def multimoora(nmtx, ncriteria):
    ratio_rank = ratio(nmtx, ncriteria, 1)[0]
    refpoint_rank = refpoint(nmtx, ncriteria, 1)[0]
    fmf_rank = fmf(nmtx, ncriteria, 1)[0]

    rank_mtx = np.vstack([ratio_rank, refpoint_rank, fmf_rank]).T

    alternatives = rank_mtx.shape[0]
    points = np.zeros(alternatives)
    for idx0, idx1 in itertools.combinations(range(alternatives), 2):
        alt0, alt1 = rank_mtx[idx0], rank_mtx[idx1]
        dom = rank.dominance(alt0, alt1)
        dom_idx = idx0 if dom > 0 else idx1
        points[dom_idx] += 1

    return rank.rankdata(points, reverse=True), rank_mtx


# =============================================================================
# OO
# =============================================================================

class RatioMOORA(DecisionMaker):
    r"""The method refers to a matrix of responses of alternatives to
    objectives, to which ratios are applied.

    In MOORA the set of ratios (by default) has the square roots of the sum
    of squared responses as denominators.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}


    These ratios, as dimensionless, seem to be the best choice among different
    ratios. These dimensionless ratios, situated between zero and one, are
    added in the case of maximization or subtracted in case of minimization:

    .. math::

        Ny_i = \sum_{i=1}^{g} Nx_{ij} - \sum_{i=1}^{g+1} Nx_{ij}

    with:
    :math:`i = 1, 2, ..., g` for the objectives to be maximized,
    :math:`i = g + 1, g + 2, ...,n` for the objectives to be minimized.

    Finally, all alternatives are ranked, according to the obtained ratios.

    Parameters
    ----------

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

    .. [1] BRAUERS, W. K.; ZAVADSKAS, Edmundas Kazimieras. The MOORA
       method and its application to privatization in a transition economy.
       Control and Cybernetics, 2006, vol. 35, p. 445-469.`

    """
    def __init__(self, wnorm="sum"):
        super(RatioMOORA, self).__init__(mnorm="vector", wnorm=wnorm)

    @doc_inherit
    def as_dict(self):
        data = super(FMFMOORA, self).as_dict()
        del data["mnorm"]
        return data

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = ratio(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


class RefPointMOORA(DecisionMaker):
    r"""Rank the alternatives from a reference point selected with the
    Min-Max Metric of Tchebycheff.

    .. math::

        \min_{j} \{ \max_{i} |r_i - x^*_{ij}| \}

    This reference point theory starts from the already normalized ratios
    as defined in the MOORA method, namely formula:

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    Preference is given to a reference point possessing as co-ordinates the
    dominating co-ordinates per attribute of the candidate alternatives and
    which is designated as the *Maximal Objective Reference Point*. This
    approach is called realistic and non-subjective as the co-ordinates,
    which are selected for the reference point, are realized in one of the
    candidate alternatives.

    Parameters
    ----------

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

           - **e_.points**: array where the i-nth element represent the
             closenees of the i-nth alternative to a reference point based
             on the *Min-Max Metric of Tchebycheff*.

    References
    ----------

    .. [1] Brauers, W. K. M., & Zavadskas, E. K. (2012). Robustness of
           MULTIMOORA: a method for multi-objective optimization.
           Informatica, 23(1), 1-25.
    .. [2] Karlin, S., & Studden, W. J. (1966). Tchebycheff systems: With
           applications in analysis and statistics. New York: Interscience.

    """
    def __init__(self, wnorm="sum"):
        super(RefPointMOORA, self).__init__(mnorm="vector", wnorm=wnorm)

    @doc_inherit
    def as_dict(self):
        data = super(FMFMOORA, self).as_dict()
        del data["mnorm"]
        return data

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = refpoint(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


class FMFMOORA(DecisionMaker):
    r"""Full Multiplicative Form, a method that is non-linear,
    non-additive, does not use weights and does not require normalization.

    To combine a minimization and maximization of different criteria
    in the same problem all the method uses the formula:

    .. math::

        U'_j = \frac{\prod_{g=1}^{i} x_{gi}}
                   {\prod_{k=i+1}^{n} x_{kj}}

    Where :math:`j` = the number of alternatives;
    :math:`i` = the number of objectives to be maximized;
    :math:`n − i` = the number of objectives to be minimize; and
    :math:`U'_j`: the utility of alternative j with objectives to be maximized
    and objectives to be minimized.

    To avoid underflow, instead the multiplication of the values we add the
    logarithms of the values; so :math:`U'_j`:, is finally defined
    as:

    .. math::

        U'_j = \sum_{g=1}^{i} \log(x_{gi}) - \sum_{k=i+1}^{n} \log(x_{kj})

    Notes
    -----

    The implementation works as follow:

    - Before determine :math:`U_j` the values  are normalized by the ratio
      sugested by MOORA.

      .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    - If we have some values of any criteria < 0 in the alternative-matrix
      we add the minimimun value of this criteria to all the criteria.
    - If we have some 0 in some criteria all the criteria is incremented by 1.
    - If some criteria is for minimization, this implementation calculates the
      inverse.
    - Instead the multiplication of the values we add the
      logarithms of the values to avoid underflow.

    Parameters
    ----------

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

    .. [1] Brauers, W. K. M., & Zavadskas, E. K. (2012). Robustness of
           MULTIMOORA: a method for multi-objective optimization.
           Informatica, 23(1), 1-25.

    """

    def __init__(self, wnorm="sum"):
        super(FMFMOORA, self).__init__(mnorm="vector", wnorm=wnorm)

    @doc_inherit
    def as_dict(self):
        data = super(FMFMOORA, self).as_dict()
        del data["mnorm"]
        return data

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
        rank, points = fmf(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


class MultiMOORA(DecisionMaker):
    r"""MULTIMOORA is compose the ranking resulting of aplyting the methods,
    RatioMOORA, RefPointMOORA and FMFMOORA.

    These three methods represent all possible methods with dimensionless
    measures in multi-objective optimization and one can not argue that one
    method is better than or is of more importance than the others; so for
    determining the final ranking the implementation maximizes how many times
    an alternative *i* dominates and alternative *j*.

    Notes
    -----

    The implementation works as follow:

    - Before determine :math:`U_j` the values  are normalized by the ratio
      sugested by MOORA.

      .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    - If we have some values of any criteria < 0 in the alternative-matrix
      we add the minimimun value of this criteria to all the criteria.
    - If we have some 0 in some criteria all the criteria is incremented by 1.
    - If some criteria is for minimization, this implementation calculates the
      inverse.
    - Instead the multiplication of the values we add the
      logarithms of the values to avoid underflow.
    - For determining the final ranking the implementation maximizes how many
      times an alternative *i* dominates and alternative *j*.

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

          - **e_.rank_mtx**: 2x3 Array where the first column is the
            RatioMOORA ranking, the second one the RefPointMOORA ranking and
            the last the FMFMOORA ranking.

    References
    ----------

    .. [1] Brauers, W. K. M., & Zavadskas, E. K. (2012). Robustness of
           MULTIMOORA: a method for multi-objective optimization.
           Informatica, 23(1), 1-25.

    """

    def __init__(self):
        super(MultiMOORA, self).__init__(mnorm="vector", wnorm="none")

    @doc_inherit
    def as_dict(self):
        data = super(MultiMOORA, self).as_dict()
        del data["wnorm"], data["mnorm"]
        return data

    @doc_inherit
    def preprocess(self, data):
        non_negative = norm.push_negatives(data.mtx, axis=0)
        non_zero = norm.add1to0(non_negative, axis=0)
        nmtx = self._mnorm(non_zero, axis=0)
        ncriteria = criteriarr(data.criteria)
        return Data(mtx=nmtx, criteria=ncriteria, weights=data.weights,
                    anames=data.anames, cnames=data.cnames)

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria = ndata.mtx, ndata.criteria
        rank, rank_mtx = multimoora(nmtx, ncriteria)
        return None, rank, {"rank_mtx": rank_mtx}
