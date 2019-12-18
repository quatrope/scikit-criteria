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

"""This module contains functions for calculate and compare ranks (ordinal
series)

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from scipy import stats


# =============================================================================
# RANKER
# =============================================================================

def rankdata(arr, reverse=False):
    """Evaluate an array and return a 1 based ranking.


    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    reverse : :py:class:`bool` default *False*
        By default (*False*) the lesser values are ranked first (like in time
        lapse in a race or Golf scoring) if is *True* the data is highest
        values are the first.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array when the i-nth element has the ranking of the i-nth element of
        the original array.


    Examples
    --------

    >>> from skcriteria.common import rank
    >>> # the fastest (the lowes value) goest first
    >>> time_laps = [0.59, 1.2, 0.3]
    >>> rank.rankdata(time_laps)
    array([2, 3, 1])
    >>> # highest is better
    >>> scores = [140, 200, 98]
    >>> rank.rankdata(scores, reverse=True)
    array([2, 1, 3])

    """
    if reverse:
        arr = np.multiply(arr, -1)
    return stats.rankdata(arr, "ordinal").astype(int)


def dominance(r0, r1):
    """Calculates the dominance or general dominance between two ranks

    Returns
    -------
    idx : integer or None
        Si es None ninguno domina a nadie, sino el indice de cual domina a cual
    dom : integer or None
        Cuantos valores domina idx al otro

    """
    N = len(r0)
    r0_lt_r1 = np.count_nonzero(np.less(r0, r1))
    if r0_lt_r1 > N - r0_lt_r1:
        return r0_lt_r1

    r1_lt_r0 = np.count_nonzero(np.less(r1, r0))
    if r1_lt_r0 > N - r1_lt_r0:
        return -r1_lt_r0

    return 0


def equality(r0, r1):
    return np.count_nonzero(np.equal(r0, r1))


def kendall_dominance(r0, r1):
    r0sum, r1sum = np.sum(r0), np.sum(r1)
    if r0sum < r1sum:
        return 0, (r0sum, r1sum)
    if r0sum > r1sum:
        return 1, (r0sum, r1sum)
    return None, (r0sum, r1sum)


def spearmanr(r0, r1):
    N = len(r0)
    num = 6.0 * np.sum(np.subtract(r0, r1) ** 2)
    denom = N * ((N ** 2) - 1)
    return 1 - (num / denom)
