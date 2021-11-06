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

from ..core import Objective, SKCTransformerABC
from ..utils import doc_inherit

# =============================================================================
# FUNCTIONS
# =============================================================================


def invert(matrix, mask):
    """Inverts all the columns selected by the mask.

    Parameters
    ----------
    matrix: :py:class:`numpy.ndarray` like.
        2D array.
    mask: :py:class:`numpy.ndarray` like.
        Boolean array like with the same elements as columns has the
        ``matrix``.

    Returns
    -------
    :py:class:`numpy.ndarray`
        New matrix with the selected columns inverted. The result matrix
        dtype float.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria import invert
        >>> invert([
        ...     [1, 2, 3],
        ...     [4, 5, 6]
        ... ],
        ... [True, False, True])
        array([[1.        , 2.        , 0.33333333],
               [0.25      , 5.        , 0.16666667]])

        >>> invert([
        ...     [1, 2, 3],
        ...     [4, 5, 6]
        ... ],
        ... [False, True, False])
        array([[1.        , 2.        , 0.33333333],
               [0.25      , 5.        , 0.16666667]])
        array([[1. , 0.5, 3. ],
               [4. , 0.2, 6. ]]

    """
    inv_mtx = np.array(matrix, dtype=float)

    inverted_values = 1.0 / inv_mtx[:, mask]
    inv_mtx[:, mask] = inverted_values

    return inv_mtx


class MinimizeToMaximize(SKCTransformerABC):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    Notes
    -----
    All the dtypes of the decision matrix are preserved except the inverted
    ones thar are converted to ``numpy.float64``.

    """

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, objectives, dtypes, **kwargs):
        # check where we need to transform
        minimize_mask = np.equal(objectives, Objective.MIN.value)

        # execute the transformation
        inv_mtx = invert(matrix, minimize_mask)

        # new objective array
        inv_objectives = np.full(
            len(objectives), Objective.MAX.value, dtype=int
        )

        # we are trying to preserve the original dtype as much as possible
        # only the minimize criteria are changed.
        inv_dtypes = np.where(minimize_mask, inv_mtx.dtype, dtypes)

        kwargs.update(
            matrix=inv_mtx, objectives=inv_objectives, dtypes=inv_dtypes
        )
        return kwargs
