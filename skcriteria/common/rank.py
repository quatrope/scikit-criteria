#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> and <nluczywo@gmail.com>
# wrote this file. As long as you retain this notice you can do whatever you
# want with this stuff. If we meet some day, and you think this stuff is worth
# it, you can buy me a WISKEY in return Juan BC and Nadia AL.


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


#==============================================================================
# DOCS
#==============================================================================

"""This module contains functions for calculate and compare ranks (ordinal
series)

"""


#==============================================================================
# IMPORTS
#==============================================================================

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
        return 0, r0_lt_r1

    r1_lt_r0 = np.count_nonzero(np.less(r1, r0))
    if r1_lt_r0 > N - r1_lt_r0:
        return 1, r1_lt_r0

    return None, 0


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
    return 1 - (num/denom)


#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    print(__doc__)
