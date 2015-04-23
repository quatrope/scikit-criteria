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

__doc__  = """Several implementations of normalization methods

"""


#==============================================================================
# IMPORTS
#==============================================================================

import numpy as np


#==============================================================================
# IMPLEMENTATIONS
#==============================================================================

def sum(arr, axis=None):
    r"""Divide all elements of the array for a summatory of the given axis. If
    *axis* is *None* sum all the array.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    This ratio method is used in various methods including
    :doc:`AHP <skcriteria.ahp>`, :doc:`Weighted Sum <skcriteria.wsum>`
    and :doc:`Weighted Product <skcriteria.wprod>`


    """
    colsum = np.sum(arr, axis=axis)
    return np.divide(arr, colsum, dtype="f")


def max(arr, axis=None):
    colmax = np.max(arr, axis=axis)
    return np.divide(arr, colmax, dtype="f")


def vector(arr, axis=None):
    r"""Calulates the set of ratios has the square roots of the sum of squared
    responses of a given axis as denominators [BRAUERS2006]_.  If *axis* is
    *None* sum all the array.

    This ratio method is used in :doc:`MOORA <skcriteria.moora>`.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}


    References
    ----------

    .. [BRAUERS2006] BRAUERS, W. K.; ZAVADSKAS, Edmundas Kazimieras. The MOORA
       method and its application to privatization in a transition economy.
       Control and Cybernetics, 2006, vol. 35, p. 445-469.`

     Examples
    --------

    >>> from skcriteria import moora
    >>>
    >>> mtx = [[1,2,3], [1,1,4], [2, 0, 1]]
    >>> criteria = [1, -1, 1]
    >>>
    >>> rnk, points = moora.ratio(mtx, criteria)
    >>>
    >>> rnk
    array([2, 1, 0])
    >>> points
    array([ 0.1021695 ,  0.74549924,  1.01261272])


    """
    sqrt = np.sqrt(np.power(arr, 2).sum(axis=axis))
    return np.divide(arr, sqrt, dtype="f")


def push_negatives(arr, axis=None):
    mins = np.min(arr, axis=axis)
    delta = (mins < 0) * mins
    return np.subtract(arr, delta)


def eps(arr, axis=None):
    eps = 0
    arr = np.asarray(arr)
    if np.any(arr == 0):
        if issubclass(arr.dtype.type, (np.inexact, float)):
            eps = np.finfo(arr.dtype.type).eps
        else:
            eps = np.finfo(np.float).eps
    return arr + eps


#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    print(__doc__)
