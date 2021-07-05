#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for scale data based o the total sum of \
values of a vector.

In addition to the main functionality, an MCDA agnostic function is offered
to scale an array along an arbitrary axis.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ..base import SKCBaseDecisionMaker, SKCMatrixAndWeightTransformerMixin
from ..utils import doc_inherit


# =============================================================================
# FUNCTIONS
# =============================================================================


def scale_by_sum(arr: np.ndarray, axis=None) -> np.ndarray:
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
    new_arr = np.array(arr, dtype=float)
    sumval = np.sum(new_arr, axis=axis, keepdims=True)
    return new_arr / sumval


class SumScaler(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    r"""Scalerbased on the total sum of values.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the total sum of all the values of that
    criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the total sum of all the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_weights)
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return scale_by_sum(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_matrix)
    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return scale_by_sum(matrix, axis=0)
