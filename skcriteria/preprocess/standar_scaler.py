#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for scale values based o the total range.

In addition to the main functionality, an agnostic function is offered
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

    new_arr = np.array(arr, dtype=float)
    mean = np.mean(new_arr, axis=axis, keepdims=True)
    std = np.std(new_arr, axis=axis, keepdims=True)
    return (new_arr - mean) / std


class StandarScaler(SKCMatrixAndWeightTransformerMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_weights)
    def transform_weights(self, weights: np.ndarray) -> np.ndarray:
        return scale_by_stdscore(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerMixin.transform_matrix)
    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return scale_by_stdscore(matrix, axis=0)
