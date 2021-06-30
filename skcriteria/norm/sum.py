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


# =============================================================================
# FUNCTIONS
# =============================================================================


def sum(arr: np.ndarray, axis=None):
    arr = np.array(arr, dtype=float)
    sumval = np.sum(arr, axis=axis, keepdims=True)
    return arr / sumval


class SumNormalizer(MatrixAndWeightNormalizerMixin, BaseDecisionMaker):
    r"""Transform all minimization criteria into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    """

    def normalize_weight(self, weights):
        return sum(weights, axis=None)

    def normalize_mtx(self, mtx):
        return sum(mtx, axis=1)
