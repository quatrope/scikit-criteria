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

__doc__ = """Several implementations of normalization methods

"""


#==============================================================================
# IMPORTS
#==============================================================================

import numpy as np


#==============================================================================
# IMPLEMENTATIONS
#==============================================================================

def sum(arr, axis=None):
    r"""Divide of every value on the array by sum of values along an
    axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    This ratio is used in various MCDA methods including
    :doc:`AHP <skcriteria.ahp>`, :doc:`Weighted Sum <skcriteria.wsum>`
    and :doc:`Weighted Product <skcriteria.wprod>`

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria.common import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> norm.sum(mtx) # ratios with the sum of the array
    aarray([[ 0.1       ,  0.2       ],
            [ 0.30000001,  0.40000001]], dtype=float32)
    >>> norm.sum(mtx, axis=0) # ratios with the sum of the array by column
    array([[ 0.25      ,  0.33333334],
           [ 0.75      ,  0.66666669]], dtype=float32)
    >>> norm.sum(mtx, axis=1) # ratios with the sum of the array by row
    array([[ 0.33333334,  0.66666669],
           [ 0.42857143,  0.5714286 ]], dtype=float32)

    """
    colsum = np.sum(arr, axis=axis, keepdims=True)
    return np.divide(arr, colsum, dtype="f")


def max(arr, axis=None):
    r"""Divide of every value on the array by max value along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria.common import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> norm.max(mtx) # ratios with the max value of the array
    array([[ 0.25,  0.5 ],
           [ 0.75,  1.  ]], dtype=float32)
    >>> norm.max(mtx, axis=0) # ratios with the max value of the arr by column
    array([[ 0.33333334,  0.5       ],
           [ 1.        ,  1.        ]], dtype=float32)
    >>> norm.max(mtx, axis=1) # ratios with the max value of the array by row
    array([[ 0.5 ,  1.  ],
           [ 0.75,  1.  ]], dtype=float32)

    """
    colmax = np.max(arr, axis=axis, keepdims=True)
    return np.divide(arr, colmax, dtype="f")


def vector(arr, axis=None):
    r"""Caculates the set of ratios as the square roots of the sum of squared
    responses of a given axis as denominators.  If *axis* is *None* sum all
    the array.

    This ratio method is used in :doc:`MOORA <skcriteria.moora>`.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria.common import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> norm.vector(mtx) # ratios with the vector value of the array
    array([[ 0.18257418,  0.36514837],
       [ 0.54772252,  0.73029673]], dtype=float32)
    >>> norm.vector(mtx, axis=0) # ratios by column
    array([[ 0.31622776,  0.44721359],
           [ 0.94868326,  0.89442718]], dtype=float32)
    >>> norm.vector(mtx, axis=1) # ratios by row
    array([[ 0.44721359,  0.89442718],
           [ 0.60000002,  0.80000001]], dtype=float32)

    """
    sqrt = np.sqrt(np.power(arr, 2).sum(axis=axis, keepdims=True))
    return np.divide(arr, sqrt, dtype="f")


def push_negatives(arr, axis=None):
    r"""If an array has negative values this function increment the values
    proportionally to made all the array positive along an axis.

    .. math::

        \overline{X}_{ij} =
            \begin{cases}
                X_{ij} + min_{X_{ij}} & \text{if } X_{ij} < 0\\
                X_{ij}          & \text{otherwise}
            \end{cases}

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria.common import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> mtx_lt0 = [[-1, 2], [3, 4]] # has a negative value
    >>> norm.push_negatives(mtx) # array without negatives don't be affected
    array([[1, 2],
           [3, 4]])
    >>> # all the array is incremented by 1 to eliminate the negative
    >>> norm.push_negatives(mtx_lt0)
    array([[0, 3],
           [4, 5]])
    >>> # by column only the first one (with the negative value) is affected
    >>> norm.push_negatives(mtx_lt0, axis=0)
    array([[0, 2],
           [4, 4]])
    >>> # by row only the first row (with the negative value) is affected
    >>> norm.push_negatives(mtx_lt0, axis=1)
    array([[0, 3],
           [3, 4]])

    """
    mins = np.min(arr, axis=axis, keepdims=True)
    delta = (mins < 0) * mins
    return np.subtract(arr, delta)


def eps(arr, axis=None):
    r"""If a value in the array is 0, then an :math:`\epsilon` is added to
    all the values

    .. math::

        \overline{X}_{ij} = X_{ij} + \epsilon

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Is not used, only exist for maintain API uniformity

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria.common import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> mtx_w0 = [[0,1], [2,3]]
    >>> norm.eps(mtx) # not afected
    array([[1, 2],
           [3, 4]])
    >>> # added a value ~0,00000000000000002, ans id only perceptible in the
    >>> # mtx_w0[0][0] element
    >>> norm.eps(mtx_w0)
    array([[  2.22044605e-16,   1.00000000e+00],
           [  2.00000000e+00,   3.00000000e+00]])

    """
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
