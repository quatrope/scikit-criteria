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


def push_negatives(arr: np.ndarray, axis=None) -> np.ndarray:
    arr = np.asarray(arr)
    mins = np.min(arr, axis=axis, keepdims=True)
    delta = (mins < 0) * mins
    return arr - delta


class PushNegatives(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_weights)
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return push_negatives(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_matrix)
    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return push_negatives(matrix, axis=0)
