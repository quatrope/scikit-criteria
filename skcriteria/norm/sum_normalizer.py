#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for inverting minimization criteria and \
converting them into maximization ones.

In addition to the main functionality, an agnostic MCDA function is offered
that inverts columns of a matrix based on a mask.

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


def sum_norm(arr: np.ndarray, axis=None) -> np.ndarray:
    r"""Divide of every value on the array by sum of values along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    Parameters
    ----------
    matrix: :py:class:`numpy.ndarray` like.
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

        >>> mtx = [[1, 2], [3, 4]]
        >>> norm.sum(mtx) # ratios with the sum of the array
        array([[ 0.1       ,  0.2       ],
               [ 0.30000001,  0.40000001]])
        >>> norm.sum(mtx, axis=0) # ratios with the sum of the array by column
        array([[ 0.25      ,  0.33333334],
               [ 0.75      ,  0.66666669]])
        >>> norm.sum(mtx, axis=1) # ratios with the sum of the array by row
        array([[ 0.33333334,  0.66666669],
               [ 0.42857143,  0.5714286 ]])

    """
    new_arr = np.array(arr, dtype=float)
    sumval = np.sum(new_arr, axis=axis, keepdims=True)
    return new_arr / sumval


class SumNormalizer(MatrixAndWeightNormalizerMixin, BaseDecisionMaker):
    r"""Transform all minimization criteria into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    """

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_weights)
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        return sum_norm(weights, axis=None)

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_matrix)
    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return sum_norm(matrix, axis=0)
