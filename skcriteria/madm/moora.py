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

from .. import norm, rank
from ..core import Data, MIN, MAX, criteriarr
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


def fmf(nmtx, criteria):
    lmtx = np.log(nmtx)

    if not np.setdiff1d(criteria, [MAX]):
        # only max
        points = np.sum(lmtx, axis=1)
    elif not np.setdiff1d(criteria, [MIN]):
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
    fmf_rank = fmf(nmtx, ncriteria)[0]

    rank_mtx = np.vstack((ratio_rank, refpoint_rank, fmf_rank)).T

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
    added in the case of maximization or subtracted in case of minimization.
    Finally, all alternatives are ranked, according to the obtained ratios.


    Parameters
    ----------

    mtx : A matrix like object
        Matrix of responses of alternatives to objectives.

    criteria : iterable
        Criteria vector. Can only containing *MIN* or *MAX* values
        with the same columns as mtx.

    weights : iterable or None
        Used to ponderate some criteria. The value of the weight is normalized
        to values between *0* and *1*. [CURCHOD2014]_


    Returns
    -------

    a Decision : (:py:class:`skcriteria.dmaker._Decision`)
        # TODO

    References
    ----------

    .. [1] BRAUERS, W. K.; ZAVADSKAS, Edmundas Kazimieras. The MOORA
       method and its application to privatization in a transition economy.
       Control and Cybernetics, 2006, vol. 35, p. 445-469.`

    """
    def __init__(self, mnorm="vector", wnorm="sum"):
        super(RatioMOORA, self).__init__(mnorm=mnorm, wnorm=wnorm)

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = ratio(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


class RefPointMOORA(DecisionMaker):

    def __init__(self, mnorm="vector", wnorm="sum"):
        super(RefPointMOORA, self).__init__(mnorm=mnorm, wnorm=wnorm)

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = refpoint(nmtx, ncriteria, nweights)
        return None, rank, {"points": points}


class FMFMOORA(DecisionMaker):

    def __init__(self, mnorm="vector"):
        super(FMFMOORA, self).__init__(mnorm=mnorm, wnorm="none")

    @doc_inherit
    def as_dict(self):
        data = super(FMFMOORA, self).as_dict()
        del data["wnorm"]
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
        rank, points = fmf(nmtx, ncriteria)
        return None, rank, {"points": points}


class MultiMOORA(DecisionMaker):

    def __init__(self, mnorm="vector"):
        super(MultiMOORA, self).__init__(mnorm=mnorm, wnorm="none")

    @doc_inherit
    def as_dict(self):
        data = super(FMFMOORA, self).as_dict()
        del data["wnorm"]
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
