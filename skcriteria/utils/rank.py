#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

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


def rank(arr, reverse=False):
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

        >>> from skcriteria.util import rank
        >>> # the fastest (the lowest value) goes first
        >>> time_laps = [0.59, 1.2, 0.3]
        >>> rank(time_laps)
        array([2, 3, 1])
        >>> # highest is better
        >>> scores = [140, 200, 98]
        >>> rank(scores, reverse=True)
        array([2, 1, 3])

    """
    if reverse:
        arr = np.multiply(arr, -1)
    return stats.rankdata(arr, "ordinal").astype(int)
