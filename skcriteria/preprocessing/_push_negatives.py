#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for remove negatives from criteria.

In addition to the main functionality, an MCDA agnostic function is offered
to push negatives values on an array along an arbitrary axis.

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


def push_negatives(arr, axis):
    r"""Increment the array until all the valuer are sean >= 0.

    If an array has negative values this function increment the values
    proportionally to made all the array positive along an axis.

    .. math::

        \overline{X}_{ij} =
            \begin{cases}
                X_{ij} + min_{X_{ij}} & \text{if } X_{ij} < 0\\
                X_{ij}          & \text{otherwise}
            \end{cases}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array with all values >= 0.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import push_negatives
        >>> mtx = [[1, 2], [3, 4]]
        >>> mtx_lt0 = [[-1, 2], [3, 4]] # has a negative value

        >>> push_negatives(mtx) # array without negatives don't be affected
        array([[1, 2],
               [3, 4]])

        # all the array is incremented by 1 to eliminate the negative
        >>> push_negatives(mtx_lt0)
        array([[0, 3],
               [4, 5]])

        # by column only the first one (with the negative value) is affected
        >>> push_negatives(mtx_lt0, axis=0)
        array([[0, 2],
               [4, 4]])
        # by row only the first row (with the negative value) is affected
        >>> push_negatives(mtx_lt0, axis=1)
        array([[0, 3],
               [3, 4]])

    """
    arr = np.asarray(arr)
    mins = np.min(arr, axis=axis, keepdims=True)
    delta = (mins < 0) * mins
    return arr - delta


class PushNegatives(SKCMatrixAndWeightTransformerABC):
    r"""Increment the matrix/weights until all the valuer are sean >= 0.

    If the matrix/weights has negative values this function increment the
    values proportionally to made all the matrix/weights positive along an
    axis.

    .. math::

        \overline{X}_{ij} =
            \begin{cases}
                X_{ij} + min_{X_{ij}} & \text{if } X_{ij} < 0\\
                X_{ij}          & \text{otherwise}
            \end{cases}

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return push_negatives(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return push_negatives(matrix, axis=0)
