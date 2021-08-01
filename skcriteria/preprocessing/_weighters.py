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

from ._distance import cenit_distance
from ..core import SKCDataValidatorMixin, SKCWeighterMixin
from ..utils import doc_inherit


# =============================================================================
# SAME WEIGHT
# =============================================================================


def equal_weights(matrix, base_value=1):
    ncriteria = np.shape(matrix)[1]
    weights = base_value / ncriteria
    return np.full(ncriteria, weights, dtype=float)


class EqualWeighter(SKCWeighterMixin):
    def __init__(self, base_value=1):
        self.base_value = base_value

    @property
    def base_value(self):
        return self._base_value

    @base_value.setter
    def base_value(self, v):
        self._base_value = float(v)

    @doc_inherit(SKCDataValidatorMixin._validate_data)
    def _validate_data(self, **kwargs):
        pass

    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return equal_weights(matrix, self.base_value)


# =============================================================================
#
# =============================================================================


def std_weights(matrix):
    std = np.std(matrix, axis=0)
    return std / np.sum(std)


class StdWeighter(SKCWeighterMixin):
    @doc_inherit(SKCDataValidatorMixin._validate_data)
    def _validate_data(self, **kwargs):
        pass

    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return std_weights(matrix)


# =============================================================================
#
# =============================================================================


def entropy_weights(matrix):
    entropy = scipy.stats.entropy(matrix, axis=0)
    return entropy / np.sum(entropy)


class EntropyWeighter(SKCWeighterMixin):
    @doc_inherit(SKCDataValidatorMixin._validate_data)
    def _validate_data(self, **kwargs):
        pass

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

    dindex = np.std(matrix, axis=0)

    corr_m1 = 1 - correlation(matrix.T)
    uweights = dindex * np.sum(corr_m1, axis=0)
    weights = uweights / np.sum(uweights)
    return weights


class Critic(SKCWeighterMixin):
    CORRELATION = {
        "pearson": pearson_correlation,
        "spearman": spearman_correlation,
    }

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
        return self._correlation

    @correlation.setter
    def correlation(self, v):
        correlation_func = self.CORRELATION.get(v, v)
        if not callable(correlation_func):
            corr_keys = ", ".join(f"'{c}'" for c in self.CORRELATION)
            raise ValueError(f"Correlation must be {corr_keys} or callable")
        self._correlation = correlation_func

    @doc_inherit(SKCDataValidatorMixin._validate_data)
    def _validate_data(self, **kwargs):
        pass

    @doc_inherit(SKCWeighterMixin._weight_matrix)
    def _weight_matrix(self, matrix, objectives, **kwargs):
        return critic_weights(
            matrix, objectives, correlation=self.correlation, scale=self.scale
        )
