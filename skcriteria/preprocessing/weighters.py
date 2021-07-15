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

import scipy.stats

# from .scalers import scale_by_ideal_point
from ..base import SKCBaseDecisionMaker, SKCWeighterMixin
from ..utils import doc_inherit


# =============================================================================
# SAME WEIGHT
# =============================================================================


def same_weight(matrix, value) -> np.ndarray:
    ncriteria = np.shape(matrix)[1]
    weights = value / ncriteria
    return np.full(ncriteria, weights, dtype=float)


class SameWeight(SKCWeighterMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix):
        return same_weight(matrix)


# =============================================================================
#
# =============================================================================


def std_weights(matrix):
    std = np.std(matrix, axis=0)
    return std / np.sum(std)


class StdWeight(SKCWeighterMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix):
        return std_weights(matrix)


# =============================================================================
#
# =============================================================================


def entropy_weights(matrix):
    entropy = scipy.stats.entropy(matrix, axis=0)
    return entropy / np.sum(entropy)


class EntropyWeights(SKCWeighterMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix):
        return entropy_weights(matrix)


# =============================================================================
#
# =============================================================================

# def critiq(mtx. scale=True):
#     mtx = scale_by_ideal_point(mtx, axis=0) if scale else np.asarray(mtx)
