#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities to add an small number (epsilon) when an array has a zero.

In addition to the main functionality, an MCDA agnostic function is offered
to add eps to zero on an array along an arbitrary axis.

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


def add_eps_to_zero(arr, eps, axis=None):
    r"""Add eps if the axis has a value 0.

    .. math::

        \overline{X}_{ij} = X_{ij} + \epsilon

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    eps: number
        Number to add if the axis has a 0.
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array with all values >= eps.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria import add_eps_to_zero
        >>> mtx = [[1, 2], [3, 4]]
        >>> mtx_w0 = [[0, 1], [2,3]]
        >>> add_eps_to_zero(mtx)
        array([[1, 2],
            [3, 4]])

        # added epsilon
        >>> add_eps_to_zero(mtx_w0, eps=0.5)
        array([[ 0.5, 1.5],
            [ 2.5, 3.5]])

    """
    arr = np.asarray(arr)
    zeros = np.any(arr == 0, axis=axis, keepdims=True)
    increment = zeros * eps
    return arr + increment


class AddEpsToZero(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    r"""Add eps if the matrix/weight whe has a value 0.

    .. math::

        \overline{X}_{ij} = X_{ij} + \epsilon

    """

    def __init__(self, eps, target):
        super().__init__(target=target)
        self.eps = eps

    @property
    def eps(self):
        """Value to add to the matrix/weight when a zero is found."""
        return self._eps

    @eps.setter
    def eps(self, eps):
        self._eps = float(eps)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_weights)
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return add_eps_to_zero(weights, eps=self.eps, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_matrix)
    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return add_eps_to_zero(matrix, eps=self.eps, axis=0)
