#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for normalize based o the range of a vector.

In addition to the main functionality, an agnostic function is offered
to normalize an array along an arbitrary axis.

"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ..base import BaseDecisionMaker, MatrixAndWeightNormalizerMixin
from ..utils import doc_inherit


# =============================================================================
# FUNCTIONS
# =============================================================================


def range_norm(arr, axis=None):
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

        >>> from skcriteria import norm
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the range of the array
        >>> norm.range_norm(mtx)
        array([[0.        , 0.33333333],
               [0.66666667, 1.        ]])

        # ratios with the range by column
        >>> norm.range_norm(mtx, axis=0)
        array([[0., 0.],
               [1., 1.]])

        # ratios with the range by row
        >>> norm.range_norm(mtx, axis=1)
        array([[0., 1.],
              [0., 1.]])

    """
    new_arr = np.array(arr, dtype=float)
    minval = np.min(new_arr, axis=axis, keepdims=True)
    maxval = np.max(new_arr, axis=axis, keepdims=True)
    return (new_arr - minval) / (maxval - minval)


class RangeNormalizer(MatrixAndWeightNormalizerMixin, BaseDecisionMaker):
    r"""Normalizer based on the range.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij} - \min{X_{ij}}}{\max_{X_{ij}} - \min_{X_{ij}}}

    If the normalizer is configured to work with 'matrix' each value
    of each criteria is divided by the range of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the range the weights.

    """

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_weights)
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        return range_norm(weights, axis=None)

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_matrix)
    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return range_norm(matrix, axis=0)
