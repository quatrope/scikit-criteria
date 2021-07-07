#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================


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
    arr = np.asarray(arr)
    zeros = np.any(arr == 0, axis=axis, keepdims=True)
    increment = zeros * eps
    return arr + increment


class AddEpsToZero(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    def __init__(self, eps, target):
        super().__init__(target=target)
        self.eps = eps

    @property
    def eps(self):
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
