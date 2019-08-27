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

"""ELECTRE is a family of multi-criteria decision analysis methods
that originated in Europe in the mid-1960s. The acronym ELECTRE stands for:
ELimination Et Choix Traduisant la REalité (ELimination and Choice Expressing
REality).

Usually the Electre Methods are used to discard some alternatives to the
problem, which are unacceptable. After that we can use another MCDA to select
the best one. The Advantage of using the Electre Methods before is that we
can apply another MCDA with a restricted set of alternatives saving much time.

"""

__all__ = ['ELECTRE1']


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import joblib

from ..validate import MAX, MIN
from ..utils.doc_inherit import doc_inherit

from ._dmaker import DecisionMaker


# =============================================================================
# CONCORDANCE
# =============================================================================

def _conc_row(idx, row, mtx, mtx_criteria, mtx_weight):
    difference = row - mtx
    outrank = (
        ((mtx_criteria == MAX) & (difference >= 0)) |
        ((mtx_criteria == MIN) & (difference <= 0))
    )
    filter_weights = mtx_weight * outrank.astype(int)
    new_row = np.sum(filter_weights, axis=1)
    return new_row


def concordance(mtx, criteria, weights, jobs=None):

    mtx_criteria = np.tile(criteria, (len(mtx), 1))
    mtx_weight = np.tile(weights, (len(mtx), 1))
    mtx_concordance = jobs(
        joblib.delayed(_conc_row)(idx, row, mtx, mtx_criteria, mtx_weight)
        for idx, row in enumerate(mtx))

    mtx_concordance = np.asarray(mtx_concordance)
    np.fill_diagonal(mtx_concordance, np.nan)
    return mtx_concordance


# =============================================================================
# DISCORDANCE
# =============================================================================

def _disc_row(idx, row, mtx, mtx_criteria, max_range):
    difference = mtx - row
    worsts = (
        ((mtx_criteria == MAX) & (difference > 0)) |
        ((mtx_criteria == MIN) & (difference < 0))
    )
    filter_difference = np.abs(difference * worsts)
    delta = filter_difference / max_range
    new_row = np.max(delta, axis=1)
    return new_row


def discordance(mtx, criteria, jobs):
    mtx_criteria = np.tile(criteria, (len(mtx), 1))
    ranges = np.max(mtx, axis=0) - np.min(mtx, axis=0)
    max_range = ranges.max()

    mtx_discordance = jobs(
        joblib.delayed(_disc_row)(idx, row, mtx, mtx_criteria, max_range)
        for idx, row in enumerate(mtx))

    mtx_discordance = np.asarray(mtx_discordance)
    np.fill_diagonal(mtx_discordance, np.nan)
    return mtx_discordance


# =============================================================================
# ELECTRE
# =============================================================================

def electre1(nmtx, ncriteria, nweights, p, q, njobs=None):
    # determine the njobs
    njobs = njobs or joblib.cpu_count()

    # get the concordance and discordance info
    # multiprocessing environment
    with joblib.Parallel(n_jobs=njobs) as jobs:
        mtx_concordance = concordance(nmtx, ncriteria, nweights, jobs)
        mtx_discordance = discordance(nmtx, ncriteria, jobs)

    with np.errstate(invalid='ignore'):
        outrank = (
            (mtx_concordance >= p) & (mtx_discordance <= q))

    kernel_mask = ~outrank.any(axis=0)
    kernel = np.where(kernel_mask)[0]
    return kernel, outrank, mtx_concordance, mtx_discordance


# =============================================================================
# OO
# =============================================================================

class ELECTRE1(DecisionMaker):
    """The ELECTRE I model  find the kernel solution in a situation where true
    criteria and restricted outranking relations are given.

    That is, ELECTRE I cannot derive the ranking of alternatives but the kernel
    set. In ELECTRE I, two indices called the concordance index and the
    discordance index are used to measure the relations between objects.

    Parameters
    ----------

    p : float, optional (default=0.65)
        Concordance threshold. Threshold of how much one alternative is at
        least as good as another to be significative.

    q : float, optional (default=0.35)
        Discordance threshold. Threshold of how much the degree one alternative
        is strictly preferred to another to be significative.

    mnorm : string, callable, optional (default="sum")
        Normalization method for the alternative matrix.

    wnorm : string, callable, optional (default="sum")
        Normalization method for the weights array.

    njobs : int, default=None
        How many cores to use to solve the linear programs and the second
        method. By default all the availables cores are used.

    Returns
    -------

    Decision : :py:class:`skcriteria.madm.Decision`
        With values:

        - **kernel_**:  Array with the indexes of the alternatives
          in he kernel.
        - **rank_**: None
        - **best_alternative_**: None
        - **alpha_solution_**: False
        - **beta_solution_**: True
        - **gamma_solution_**: False
        - **e_**: Particular data created by this method.

          - **e_.outrank**: numpy.ndarray of bool
            The outranking matrix of superation. If the element[i][j] is True
            The alternative ``i`` outrank the alternative ``j``.
          - **e_.mtx_concordance**: numpy.ndarray
            The concordance indexes matrix where the element[i][j] measures how
            much the alternative ``i`` is at least as good as ``j``.
          - **e_.mtx_discordance**: numpy.ndarray
            The discordance indexes matrix where the element[i][j] measures the
            degree to which the alternative ``i`` is strictly
            preferred to ``j``.
          - **e_.p**: float
            Concordance index threshold.
          - **e_.q**: float
            Discordance index threshold.

    References
    ----------

    .. [1] Roy, B. (1990). The outranking approach and the foundations of
       ELECTRE methods. In Readings in multiple criteria decision aid
       (pp.155-183). Springer, Berlin, Heidelberg.
    .. [2] Roy, B. (1968). Classement et choix en présence de points de vue
       multiples. Revue française d'informatique et de recherche
       opérationnelle, 2(8), 57-75.
    .. [3] Tzeng, G. H., & Huang, J. J. (2011). Multiple attribute decision
       making: methods and applications. CRC press.


    """

    def __init__(self, p=.65, q=.35, mnorm="sum", wnorm="sum", njobs=None):
        super(ELECTRE1, self).__init__(mnorm=mnorm, wnorm=wnorm)
        self._p = float(p)
        self._q = float(q)
        self._njobs = njobs

    @doc_inherit
    def as_dict(self):
        base = super(ELECTRE1, self).as_dict()
        base.update({"p": self._p, "q": self._q})
        return base

    @doc_inherit
    def solve(self, ndata):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        kernel, outrank, mtx_concordance, mtx_discordance = electre1(
            nmtx=nmtx, ncriteria=ncriteria, nweights=nweights,
            p=self._p, q=self._q)

        extra = {
            "outrank": outrank,
            "mtx_concordance": mtx_concordance,
            "mtx_discordance": mtx_discordance,
            "p": self.p, "q": self.q}

        return kernel, None, extra

    @property
    def p(self):
        """Concordance threshold. Threshold of how much one alternative is at
        least as good as another to be significative.

        """
        return self._p

    @property
    def q(self):
        """Discordance threshold. Threshold of how much the degree one
        alternative is strictly preferred to another to be significative.

        """
        return self._q

    @property
    def njobs(self):
        """How many cores to use to solve the linear programs and the second
        method. By default all the availables cores are used.

        """
        return self._njobs
