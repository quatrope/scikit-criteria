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

__doc__ = """Several implementations of normalization methods

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from numpy import linalg


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DuplicatedNameError(ValueError):
    pass


class NormalizerNotFound(AttributeError):
    pass


class FunctionNotRegisteredAsNormalizer(ValueError):
    pass


# =============================================================================
# REGISTERS
# =============================================================================

NORMALIZERS = {}


def register(name, func=None):
    if name in NORMALIZERS:
        raise DuplicatedNameError(name)
    if func is None:
        def _dec(func):
            NORMALIZERS[name] = func
            return func
        return _dec
    else:
        NORMALIZERS[name] = func
        return func


def get(name):
    try:
        return NORMALIZERS[name]
    except KeyError:
        raise NormalizerNotFound(name)


def nameof(normalizer):
    for k, v in NORMALIZERS.items():
        if v == normalizer:
            return k
    raise FunctionNotRegisteredAsNormalizer(str(normalizer))


def norm(name, arr, axis=None):
    normalizer = get(name)
    return normalizer(arr, axis=axis)


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================

@register("none")
def none(arr, axis=None):
    """This do not nothing and only try to return an numpy.ndarray
    of the given data

    """
    return np.asarray(arr)


@register("sum")
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
            [ 0.30000001,  0.40000001]], dtype=float64)
    >>> norm.sum(mtx, axis=0) # ratios with the sum of the array by column
    array([[ 0.25      ,  0.33333334],
           [ 0.75      ,  0.66666669]], dtype=float64)
    >>> norm.sum(mtx, axis=1) # ratios with the sum of the array by row
    array([[ 0.33333334,  0.66666669],
           [ 0.42857143,  0.5714286 ]], dtype=float64)

    """
    arr = np.asarray(arr, dtype=float)
    sumval = np.sum(arr, axis=axis, keepdims=True)
    return arr / sumval


@register("max")
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
           [ 0.75,  1.  ]], dtype=float64)
    >>> norm.max(mtx, axis=0) # ratios with the max value of the arr by column
    array([[ 0.33333334,  0.5       ],
           [ 1.        ,  1.        ]], dtype=float64)
    >>> norm.max(mtx, axis=1) # ratios with the max value of the array by row
    array([[ 0.5 ,  1.  ],
           [ 0.75,  1.  ]], dtype=float64)

    """
    arr = np.asarray(arr, dtype=float)
    maxval = np.max(arr, axis=axis, keepdims=True)
    return arr / maxval


@register("vector")
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
       [ 0.54772252,  0.73029673]], dtype=float64)
    >>> norm.vector(mtx, axis=0) # ratios by column
    array([[ 0.31622776,  0.44721359],
           [ 0.94868326,  0.89442718]], dtype=float64)
    >>> norm.vector(mtx, axis=1) # ratios by row
    array([[ 0.44721359,  0.89442718],
           [ 0.60000002,  0.80000001]], dtype=float64)

    """
    arr = np.asarray(arr, dtype=float)
    frob = linalg.norm(arr, None, axis=axis)
    return arr / frob


@register("push_negatives")
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
    arr = np.asarray(arr)
    mins = np.min(arr, axis=axis, keepdims=True)
    delta = (mins < 0) * mins
    return arr - delta


@register("add1to0")
def add1to0(arr, axis=None):
    r"""If a value in the array is 0, then an :math:`1` is added to
    all the values

    .. math::

        \overline{X}_{ij} = X_{ij} + 1

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
    >>> mtx_w0 = [[0,1], [2,3]]
    >>> norm.add1to0(mtx)
    array([[1, 2],
           [3, 4]])
    >>> # added 1
    >>> norm.add1to0(mtx_w0)
    array([[  1, 2],
           [  3, 4]])

    """
    arr = np.asarray(arr)
    if 0 in arr:
        if len(arr.shape) == 1 or axis is None:
            return arr + 1
        else:
            zeros = np.any(arr == 0, axis=axis)
            increment = np.zeros(zeros.shape[0])
            increment[zeros] = 1
            return arr + increment
    return arr
