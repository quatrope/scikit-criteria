#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functions for calculate and compare ranks (ordinal series)."""


# =============================================================================
# IMPORTS
# =============================================================================

from collections import namedtuple

import numpy as np

from scipy import stats

# =============================================================================
# RANKER
# =============================================================================


def rank_values(arr, reverse=False):
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
    :py:class:`numpy.ndarray`
        Array of rankings the i-nth element has the ranking of the i-nth
        element of the row array.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.util.rank import rank_values
        >>> # the fastest (the lowest value) goes first
        >>> time_laps = [0.59, 1.2, 0.3]
        >>> rank_values(time_laps)
        array([2, 3, 1])
        >>> # highest is better
        >>> scores = [140, 200, 98]
        >>> rank_values(scores, reverse=True)
        array([2, 1, 3])

    """
    if reverse:
        arr = np.multiply(arr, -1)
    return stats.rankdata(arr, "ordinal").astype(int)


# =============================================================================
# DOMINANCE
# =============================================================================

_Dominance = namedtuple(
    "dominance",
    ["eq", "aDb", "bDa", "eq_where", "aDb_where", "bDa_where"],
)


def dominance(array_a, array_b, reverse=False):
    """Calculate the dominance or general dominance between two arrays.

    Parameters
    ----------
    array_a:
        The first array to compare.
    array_b:
        The second array to compare.
    reverse: bool (default=False)
        array_a[i] ≻ array_b[i] if array_a[i] > array_b[i] if reverse
        is False, otherwise array_a[i] ≻ array_b[i] if array_a[i] < array_b[i].

    Returns
    -------
    dominance: _Dominance
        Named tuple with 4 parameters:

        - eq: How many values are equals in both arrays.
        - aDb: How many values of array_a dominate those of the same
            position in array_b.
        - bDa: How many values of array_b dominate those of the same
            position in array_a.
        - eq_where: Where the values of array_a are equals those of the same
            position in array_b.
        - aDb_where: Where the values of array_a dominates those of the same
            position in array_b.
        - bDa_where: Where the values of array_b dominates those of the same
            position in array_a.

    """
    if np.shape(array_a) != np.shape(array_b):
        raise ValueError("array_a and array_b must be of the same shape")

    domfunc = np.less if reverse else np.greater

    array_a = np.asarray(array_a, dtype=int)
    array_b = np.asarray(array_b, dtype=int)

    eq_where = array_a == array_b
    aDb_where = domfunc(array_a, array_b)
    bDa_where = domfunc(array_b, array_a)

    return _Dominance(
        # resume
        eq=np.sum(eq_where),
        aDb=np.sum(aDb_where),
        bDa=np.sum(bDa_where),
        # locations
        eq_where=eq_where,
        aDb_where=aDb_where,
        bDa_where=bDa_where,
    )
