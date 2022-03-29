#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for inverting minimization criteria and \
converting them into maximization ones.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ..core import Objective, SKCTransformerABC
from ..utils import deprecated, doc_inherit

# =============================================================================
# FUNCTIONS
# =============================================================================


class SKCObjectivesInverterABC(SKCTransformerABC):

    _skcriteria_abstract_class = True

    def _invert(self, matrix, minimize_mask):
        raise NotImplementedError()

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, objectives, dtypes, **kwargs):
        # check where we need to transform
        minimize_mask = np.equal(objectives, Objective.MIN.value)

        # execute the transformation
        inv_mtx = self._invert(matrix, minimize_mask)

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


class NegateMinimize(SKCObjectivesInverterABC):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{-{C}}`.

    """
    _skcriteria_parameters = []

    @doc_inherit(SKCObjectivesInverterABC._invert)
    def _invert(self, matrix, minimize_mask):
        inv_mtx = np.array(matrix, dtype=float)

        inverted_values = -inv_mtx[:, minimize_mask]
        inv_mtx[:, minimize_mask] = inverted_values

        return inv_mtx


class InvertMinimize(SKCObjectivesInverterABC):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    Notes
    -----
    All the dtypes of the decision matrix are preserved except the inverted
    ones thar are converted to ``numpy.float64``.

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCObjectivesInverterABC._invert)
    def _invert(self, matrix, minimize_mask):
        inv_mtx = np.array(matrix, dtype=float)

        inverted_values = 1.0 / inv_mtx[:, minimize_mask]
        inv_mtx[:, minimize_mask] = inverted_values

        return inv_mtx


@deprecated(
    reason="Use 'skcriteria.preprocessing.InvertMinimize' instead",
    version=0.7,
)
@doc_inherit(InvertMinimize)
class MinimizeToMaximize(InvertMinimize):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    Notes
    -----
    All the dtypes of the decision matrix are preserved except the inverted
    ones thar are converted to ``numpy.float64``.

    """
