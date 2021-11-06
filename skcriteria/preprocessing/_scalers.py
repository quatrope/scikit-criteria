#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for scale values based on differrent strategies.

In addition to the Transformers, a collection of an MCDA agnostic functions
are offered to scale an array along an arbitrary axis.

"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
from numpy import linalg

from ..core import SKCMatrixAndWeightTransformerABC
from ..utils import doc_inherit

# =============================================================================
# STANDAR SCALER
# =============================================================================


def scale_by_stdscore(arr, axis=None):
    r"""Standardize the values by removing the mean and divided by the std-dev.

    The standard score of a sample `x` is calculated as:

    .. math::

        z = (x - \mu) / \sigma

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_stdscore
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the max value of the array
        >>> scale_by_stdscore(mtx)
        array([[-1.34164079, -0.4472136 ],
               [ 0.4472136 ,  1.34164079]])

        # ratios with the max value of the arr by column
        >>> scale_by_stdscore(mtx, axis=0)
        array([[-1., -1.],
               [ 1.,  1.]])

        # ratios with the max value of the array by row
        >>> scale_by_stdscore(mtx, axis=1)
        array([[-1.,  1.],
               [-1.,  1.]])

    """
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    return (arr - mean) / std


class StandarScaler(SKCMatrixAndWeightTransformerABC):
    """Standardize the dm by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the values, and `s` is the standard deviation
    of the training samples or one if `with_std=False`.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_stdscore(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_stdscore(matrix, axis=0)


# =============================================================================
# VECTOR SCALER
# =============================================================================


def scale_by_vector(arr, axis=None):
    r"""Divide the array by norm of values defined vector along an axis.

    Calculates the set of ratios as the square roots of the sum of squared
    responses of a given axis as denominators.  If *axis* is *None* sum all
    the array.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_vector
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the vector value of the array
        >>> scale_by_vector(mtx)
        array([[ 0.18257418,  0.36514837],
               [ 0.54772252,  0.73029673]])

        # ratios by column
        >>> scale_by_vector(mtx, axis=0)
        array([[ 0.31622776,  0.44721359],
               [ 0.94868326,  0.89442718]])

        # ratios by row
        >>> scale_by_vector(mtx, axis=1)
        array([[ 0.44721359,  0.89442718],
               [ 0.60000002,  0.80000001]])

    """
    arr = np.asarray(arr, dtype=float)
    frob = linalg.norm(arr, None, axis=axis)
    return arr / frob


class VectorScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scaler based on the norm of the vector..

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the norm of the vector defined by the values
    of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the vector defined by the values
    of the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_vector(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_vector(matrix, axis=0)


# =============================================================================
# MINMAX
# =============================================================================


def scale_by_minmax(arr, axis=None):
    r"""Fraction of the range normalizer.

    Subtracts to each value of the array the minimum and then divides
    it by the total range.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij} - \min{X_{ij}}}{\max_{X_{ij}} - \min_{X_{ij}}}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios


    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_minmax
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the range of the array
        >>> scale_by_minmax(mtx)
        array([[0.        , 0.33333333],
               [0.66666667, 1.        ]])

        # ratios with the range by column
        >>> scale_by_minmax(mtx, axis=0)
        array([[0., 0.],
               [1., 1.]])

        # ratios with the range by row
        >>> scale_by_minmax(mtx, axis=1)
        array([[0., 1.],
              [0., 1.]])

    """
    arr = np.asarray(arr, dtype=float)
    minval = np.min(arr, axis=axis, keepdims=True)
    maxval = np.max(arr, axis=axis, keepdims=True)
    return (arr - minval) / (maxval - minval)


class MinMaxScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scaler based on the range.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij} - \min{X_{ij}}}{\max_{X_{ij}} - \min_{X_{ij}}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the range of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the range the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_minmax(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_minmax(matrix, axis=0)


# =============================================================================
# SUM
# =============================================================================


def scale_by_sum(arr, axis=None):
    r"""Divide of every value on the array by sum of values along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_sum
        >>> mtx = [[1, 2], [3, 4]]

        >>> scale_by_sum(mtx) # ratios with the sum of the array
        array([[ 0.1       ,  0.2       ],
               [ 0.30000001,  0.40000001]])

        # ratios with the sum of the array by column
        >>> scale_by_sum(mtx, axis=0)
        array([[ 0.25      ,  0.33333334],
               [ 0.75      ,  0.66666669]])

        # ratios with the sum of the array by row
        >>> scale_by_sum(mtx, axis=1)
        array([[ 0.33333334,  0.66666669],
               [ 0.42857143,  0.5714286 ]])

    """
    arr = np.asarray(arr, dtype=float)
    sumval = np.sum(arr, axis=axis, keepdims=True)
    return arr / sumval


class SumScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scalerbased on the total sum of values.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the total sum of all the values of that
    criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the total sum of all the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_sum(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_sum(matrix, axis=0)


# =============================================================================
# MAX
# =============================================================================


def scale_by_max(arr, axis=None):
    r"""Divide of every value on the array by max value along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_max
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the max value of the array
        >>> scale_by_max(mtx)
        array([[ 0.25,  0.5 ],
               [ 0.75,  1.  ]])

        # ratios with the max value of the arr by column
        >>> scale_by_max(mtx, axis=0)
        array([[ 0.33333334,  0.5],
               [ 1.        ,  1. ]])

        # ratios with the max value of the array by row
        >>> scale_by_max(mtx, axis=1)
        array([[ 0.5 ,  1.],
               [ 0.75,  1.]])

    """
    arr = np.asarray(arr, dtype=float)
    maxval = np.max(arr, axis=axis, keepdims=True)
    return arr / maxval


class MaxScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scaler based on the maximum values.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the maximum value of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the maximum value the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_max(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_max(matrix, axis=0)
