#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities to add an value when an array has a zero.

In addition to the main functionality, an MCDA agnostic function is offered
to add value to zero on an array along an arbitrary axis.

"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ..core import SKCMatrixAndWeightTransformerABC
from ..utils import doc_inherit

# =============================================================================
# FUNCTIONS
# =============================================================================


def add_value_to_zero(arr, value, axis=None):
    r"""Add value if the axis has a value 0.

    .. math::

        \overline{X}_{ij} = X_{ij} + value

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    value: number
        Number to add if the axis has a 0.
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array with all values >= value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria import add_to_zero

        # no zero
        >>> mtx = [[1, 2], [3, 4]]
        >>> add_to_zero(mtx, value=0.5)
        array([[1, 2],
               [3, 4]])

        # with zero
        >>> mtx = [[0, 1], [2,3]]
        >>> add_to_zero(mtx, value=0.5)
        array([[ 0.5, 1.5],
               [ 2.5, 3.5]])

    """
    arr = np.asarray(arr)
    zeros = np.any(arr == 0, axis=axis, keepdims=True)
    increment = zeros * value
    return arr + increment


class AddValueToZero(SKCMatrixAndWeightTransformerABC):
    r"""Add value if the matrix/weight whe has a value 0.

    .. math::

        \overline{X}_{ij} = X_{ij} + value

    """

    def __init__(self, value, target):
        super().__init__(target=target)
        self.value = value

    @property
    def value(self):
        """Value to add to the matrix/weight when a zero is found."""
        return self._eps

    @value.setter
    def value(self, value):
        self._eps = float(value)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return add_value_to_zero(weights, value=self.value, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return add_value_to_zero(matrix, value=self.value, axis=0)
