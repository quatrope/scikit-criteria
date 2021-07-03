#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for normalize based o the max value of a vector.

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


def max_norm(arr, axis=None):
    r"""Divide of every value on the array by max value along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    Parameters
    ----------
    arr : (:py:class:`numpy.ndarray`, :py:class:`numpy.ndarray`)
        A array with values

    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria import norm
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the max value of the array
        >>> norm.max_norm(mtx)
        array([[ 0.25,  0.5 ],
            [ 0.75,  1.  ]])

        # ratios with the max value of the arr by column
        >>> norm.max_norm(mtx, axis=0)
        array([[ 0.33333334,  0.5],
            [ 1.        ,  1. ]])

        # ratios with the max value of the array by row
        >>> norm.max_norm(mtx, axis=1)
        array([[ 0.5 ,  1.],
            [ 0.75,  1.]])

    """
    new_arr = np.array(arr, dtype=float)
    maxval = np.max(new_arr, axis=axis, keepdims=True)
    return new_arr / maxval


class MaxNormalizer(MatrixAndWeightNormalizerMixin, BaseDecisionMaker):
    r"""Normalizer based on the maximum values.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    If the normalizer is configured to work with 'matrix' each value
    of each criteria is divided by the maximum value of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the maximum value the weights.

    """

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_weights)
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        return max_norm(weights, axis=None)

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_matrix)
    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return max_norm(matrix, axis=0)
