#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for scale values based on the normal score.

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


class StandarScaler(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    """Standardize the dm by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the values, and `s` is the standard deviation
    of the training samples or one if `with_std=False`.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_weights)
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return scale_by_stdscore(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_matrix)
    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return scale_by_stdscore(matrix, axis=0)
