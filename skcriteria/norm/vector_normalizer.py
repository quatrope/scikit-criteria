#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for normalize based o the total norm of \
vector defined by values on an array..

In addition to the main functionality, an agnostic function is offered
to normalize an array along an arbitrary axis.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
from numpy import linalg

from ..base import BaseDecisionMaker, MatrixAndWeightNormalizerMixin
from ..utils import doc_inherit


# =============================================================================
# FUNCTIONS
# =============================================================================


def vector_norm(arr: np.ndarray, axis=None) -> np.ndarray:
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

        >>> from skcriteria import norm
        >>> mtx = [[1, 2], [3, 4]]
        >>> norm.vector(mtx) # ratios with the vector value of the array
        array([[ 0.18257418,  0.36514837],
               [ 0.54772252,  0.73029673]])
        >>> norm.vector(mtx, axis=0) # ratios by column
        array([[ 0.31622776,  0.44721359],
            [ 0.94868326,  0.89442718]])
        >>> norm.vector(mtx, axis=1) # ratios by row
        array([[ 0.44721359,  0.89442718],
            [ 0.60000002,  0.80000001]])

    """
    new_arr = np.array(arr, dtype=float)
    frob = linalg.norm(new_arr, None, axis=axis)
    return new_arr / frob


class VectorNormalizer(MatrixAndWeightNormalizerMixin, BaseDecisionMaker):
    r"""Normalizer based on the norm of the vector..

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    If the normalizer is configured to work with 'matrix' each value
    of each criteria is divided by the norm of the vector defined by the values
    of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the vector defined by the values
    of the weights.

    """

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_weights)
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        return vector_norm(weights, axis=None)

    @doc_inherit(MatrixAndWeightNormalizerMixin.normalize_matrix)
    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return vector_norm(matrix, axis=0)
