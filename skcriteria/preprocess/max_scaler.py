#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for scale data based o the max value of a vector.

In addition to the main functionality, an MCDA agnostic function is offered
to scale data on an array along an arbitrary axis.

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


class MaxScaler(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    r"""Scaler based on the maximum values.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\max_{X_{ij}}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the maximum value of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the maximum value the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_weights)
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return scale_by_max(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_matrix)
    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return scale_by_max(matrix, axis=0)
