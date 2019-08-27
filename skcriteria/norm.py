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

"""Several implementations of normalization methods

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from numpy import linalg

from .validate import MIN, MAX, criteriarr


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


def get(name, d=None):
    try:
        return NORMALIZERS[name]
    except KeyError:
        if d is not None:
            return d
        raise NormalizerNotFound(name)


def nameof(normalizer):
    for k, v in NORMALIZERS.items():
        if v == normalizer:
            return k
    raise FunctionNotRegisteredAsNormalizer(str(normalizer))


def norm(name, arr, *args, **kwargs):
    normalizer = get(name)
    return normalizer(arr, *args, **kwargs)


# =============================================================================
# IMPLEMENTATIONS
# =============================================================================

@register("none")
def none(arr, criteria=None, axis=None):
    """This do not nothing and only try to return an numpy.ndarray
    of the given data

    """
    return np.asarray(arr)


@register("sum")
def sum(arr, criteria=None, axis=None):
    r"""Divide of every value on the array by sum of values along an
    axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    criteria : Not used

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria import norm
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
def max(arr, criteria=None, axis=None):
    r"""Divide of every value on the array by max value along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    criteria : Not used

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria import norm
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
def vector(arr, criteria=None, axis=None):
    r"""Caculates the set of ratios as the square roots of the sum of squared
    responses of a given axis as denominators.  If *axis* is *None* sum all
    the array.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    criteria : Not used

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria import norm
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
def push_negatives(arr, criteria=None, axis=None):
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

    criteria : Not used

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria import norm
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
def add1to0(arr, criteria=None, axis=None):
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

    criteria : Not used

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria import norm
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
            increment = np.zeros(zeros.shape)
            increment[zeros] = 1
            return arr + increment
    return arr


@register("addepsto0")
def addepsto0(arr, criteria=None, axis=None):
    r"""If a value in the array is 0, then an :math:`\epsilon` is
    added to all the values

    .. math::

        \overline{X}_{ij} = X_{ij} + \epsilon

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    criteria : Not used

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios

    Examples
    --------

    >>> from skcriteria import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> mtx_w0 = [[0, 1], [2,3]]
    >>> norm.addepsto0(mtx)
    array([[1, 2],
           [3, 4]])
    >>> # added epsilon
    >>> norm.addepsto0(mtx_w0)
    array([[  2.22e-16, 1],
           [         2, 3]])

    """
    arr = np.asarray(arr)
    if 0 in arr:
        arr_type = arr.dtype.type
        if not issubclass(arr_type, (np.floating, float)):
            arr_type = float
        eps = np.finfo(arr_type).eps
        if len(arr.shape) == 1 or axis is None:
            return arr + eps
        else:
            zeros = np.any(arr == 0, axis=axis)
            increment = np.zeros(zeros.shape[0])
            increment[zeros] = eps
            return arr + increment
    return arr


@register("ideal_point")
def ideal_point(arr, criteria=None, axis=None):
    """This transformation is based on the concept of the ideal
    point. So, the value :math:`x_{aj}` below, expresses the degree to which
    the  alternative a is close to the ideal value :math:`f_j^*`, which is the
    best performance in criterion :math:`j`, and far from the anti-ideal
    value :math:`f_{j^*}`., which is the worst performance in criterion
    :math:`j`. Both :math:`f_j^*` and :math:`f_{j^*}`, are achieved by at
    least one of the alternatives under consideration.

    .. math::

        x_{aj} = \\frac{ f_j(a) - f_{j^*} }{ f_j^* - f_{j^*} }

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    criteria : :py:class:`numpy.ndarray`
        Criteria array to determine the ideal and nadir points of every
        criteria.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios


    Examples
    --------

    >>> from skcriteria import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> norm.ideal_point(mtx, axis=0)
    array([[ 0.,  0.],
           [ 1.,  1.]])

    """

    if criteria is None:
        raise TypeError("you must provide a criteria")

    if axis not in (0, 1, None):
        msg = "'axis' must be 0, 1 or None. Found: {}"
        raise ValueError(msg.format(axis))

    arr = np.asarray(arr, dtype=float)
    criteria = criteriarr(criteria)
    if axis is None:
        if len(set(criteria)) != 1:
            msg = "If 'axis' is None all the 'criteria' must be the same"
            raise ValueError(msg)
        criteria = criteria[0]
        idealf, nadirf = (
            (np.max, np.min)
            if criteria == MAX
            else (np.min, np.max))
        ideal, nadir = idealf(arr), nadirf(arr)
    elif axis == 1:
        arr = arr.T

    maxs = np.max(arr, axis=0)
    mins = np.min(arr, axis=0)

    ideal = np.where(criteria == MAX, maxs, mins)
    nadir = np.where(criteria == MAX, mins, maxs)

    result = (arr - nadir) / (ideal - nadir)

    if axis == 1:
        result = result.T

    return result


@register("invert_min")
def invert_min(arr, criteria=None, axis=None):
    """Invert all the axis whith minimizartion criteria

    Parameters
    ----------

    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    criteria : :py:class:`numpy.ndarray`
        Criteria array.

    Returns
    -------

    narray : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        array of ratios


    Examples
    --------

    >>> from skcriteria import norm
    >>> mtx = [[1, 2], [3, 4]]
    >>> norm.ideal_point(mtx, criteria=[1, -1], axis=0)
    array([[ 1.,   0.5],
           [ 3.,  0.25]])

    """

    if criteria is None:
        raise TypeError("you must provide a criteria")

    if axis not in (0, 1, None):
        msg = "'axis' must be 0, 1 or None. Found: {}"
        raise ValueError(msg.format(axis))

    arr = np.asarray(arr, dtype=float)
    criteria = criteriarr(criteria)

    if axis is None and len(set(criteria)) != 1:
        msg = "If 'axis' is None all the 'criteria' must be the same"
        raise ValueError(msg)
    elif axis == 1:
        arr = arr.T

    if MIN in criteria:
        mincrits = np.squeeze(np.where(criteria == MIN))

        if np.ndim(arr) == 1:
            mincrits_inverted = 1.0 / arr[mincrits]
            arr = arr.astype(mincrits_inverted.dtype.type)
            arr[mincrits] = mincrits_inverted
        else:
            mincrits_inverted = 1.0 / arr[:, mincrits]
            arr = arr.astype(mincrits_inverted.dtype.type)
            arr[:, mincrits] = mincrits_inverted

    if axis == 1:
        arr = arr.T

    return arr
