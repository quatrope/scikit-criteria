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

from ..base import SKCBaseDecisionMaker, SKCWeighterMixin
from ..utils import doc_inherit
from .distance import cenit_distance

# =============================================================================
# SAME WEIGHT
# =============================================================================


def equal_weights(matrix, base_value=1):
    ncriteria = np.shape(matrix)[1]
    weights = base_value / ncriteria
    return np.full(ncriteria, weights, dtype=float)


class EqualWeighter(SKCWeighterMixin, SKCBaseDecisionMaker):
    def __init__(self, base_value=1):
        self.base_value = base_value

    @property
    def base_value(self):
        return self._base_value

    @base_value.setter
    def base_value(self, v):
        self._base_value = float(v)

    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return equal_weights(matrix, self.base_value)


# =============================================================================
#
# =============================================================================


def std_weights(matrix):
    std = np.std(matrix, axis=0)
    return std / np.sum(std)


class StdWeighter(SKCWeighterMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return std_weights(matrix)


# =============================================================================
#
# =============================================================================


def entropy_weights(matrix):
    entropy = scipy.stats.entropy(matrix, axis=0)
    return entropy / np.sum(entropy)


class EntropyWeighter(SKCWeighterMixin, SKCBaseDecisionMaker):
    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return entropy_weights(matrix)


# =============================================================================
#
# =============================================================================


def pearson_correlation(arr):
    return np.corrcoef(arr)


def spearman_correlation(arr):
    return scipy.stats.spearmanr(arr.T, axis=0).correlation


def critic_weights(
    matrix, objectives, correlation=pearson_correlation, scale=True
):
    matrix = np.asarray(matrix, dtype=float)
    matrix = cenit_distance(matrix, objectives=objectives) if scale else matrix

    dindex = np.std(matrix)
    corr_m1 = 1 - correlation(matrix.T)
    uweights = dindex * np.sum(corr_m1, axis=0)
    weights = uweights / np.sum(uweights)
    return weights


class Critic(SKCWeighterMixin, SKCBaseDecisionMaker):
    def __init__(self, correlation="pearson", scale=True):
        self.correlation = correlation
        self.scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, v):
        self._scale = bool(v)

    @property
    def correlation(self):
        return self.correlation

    @correlation.setter
    def correlation(self, v):
        if v == "pearson":
            self._correlation = pearson_correlation
        elif v == "spearman":
            self._correlation = spearman_correlation
        elif callable(v):
            self._correlation = v
        else:
            raise TypeError(
                "correlation must be 'pearson', 'spearmen' or callable"
            )

    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, objectives, **kwargs):
        return critic_weights(
            matrix, objectives, correlation=self.correlation, scale=self.scale
        )
