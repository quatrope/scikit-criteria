#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Multi-objective optimization on the basis of ratio analysis methods."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..base import SKCRankerMixin
from ..data import Objective, RankResult
from ..utils import doc_inherit, rank

# =============================================================================
# Ratio MOORA
# =============================================================================


def ratio(matrix, objectives, weights):

    # invert the minimization criteria
    # If we multiply by -1 (min) the weights,
    # when we multipliying this weights by the matrix we emulate
    # the -+ ratio mora strategy
    objective_x_weights = weights * objectives

    # calculate raning by inner prodcut
    rank_mtx = np.inner(matrix, objective_x_weights)
    score = np.squeeze(np.asarray(rank_mtx))
    return rank(score, reverse=True), score


class RatioMOORA(SKCRankerMixin):
    @doc_inherit(SKCRankerMixin._validate_data)
    def _validate_data(self, **kwargs):
        ...

    @doc_inherit(SKCRankerMixin._rank_data)
    def _rank_data(self, matrix, objectives, weights, **kwargs):
        rank, score = ratio(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCRankerMixin._make_result)
    def _make_result(self, anames, rank, extra):
        return RankResult("RatioMOORA", anames=anames, rank=rank, extra=extra)


# =============================================================================
# Reference Point Moora
# =============================================================================


def refpoint(matrix, objectives, weights):

    # max and min reference points
    rpmax = np.max(matrix, axis=0)
    rpmin = np.min(matrix, axis=0)

    # merge two reference points acoording objectives
    mask = np.where(objectives == Objective.MAX.value, objectives, 0)
    reference_point = np.where(mask, rpmax, rpmin)

    # create rank matrix
    rank_mtx = np.max(np.abs(weights * (matrix - reference_point)), axis=1)
    score = np.squeeze(np.asarray(rank_mtx))
    return rank(score), score, reference_point


class ReferencePointMOORA(SKCRankerMixin):
    @doc_inherit(SKCRankerMixin._validate_data)
    def _validate_data(self, **kwargs):
        ...

    @doc_inherit(SKCRankerMixin._rank_data)
    def _rank_data(self, matrix, objectives, weights, **kwargs):
        rank, score, reference_point = refpoint(matrix, objectives, weights)
        return rank, {"score": score, "reference_point": reference_point}

    @doc_inherit(SKCRankerMixin._make_result)
    def _make_result(self, anames, rank, extra):
        return RankResult(
            "ReferencePointMOORA", anames=anames, rank=rank, extra=extra
        )


# =============================================================================
# FULL MULTIPLICATIVE FORM
# =============================================================================


def fmf(matrix, objectives, weights):

    weighted_matrix = np.log(np.multiply(matrix, weights))

    if Objective.MAX.value in objectives:
        max_columns = weighted_matrix[:, objectives == Objective.MAX.value]
        Aj = np.sum(max_columns, axis=1)
    else:
        Aj = 1.0

    if Objective.MIN.value in objectives:
        min_columns = weighted_matrix[:, objectives == Objective.MIN.value]
        Bj = np.sum(min_columns, axis=1)
    else:
        Bj = 0.0

    score = Aj - Bj

    return rank(score, reverse=True), score


class FullMultiplicativeForm(SKCRankerMixin):
    @doc_inherit(SKCRankerMixin._validate_data)
    def _validate_data(self, matrix, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError(
                "FullMultiplicativeForm can't operate with values <= 0"
            )

    @doc_inherit(SKCRankerMixin._rank_data)
    def _rank_data(self, matrix, objectives, weights, **kwargs):
        rank, score = fmf(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCRankerMixin._make_result)
    def _make_result(self, anames, rank, extra):
        return RankResult(
            "FullMultiplicativeForm",
            anames=anames,
            rank=rank,
            extra=extra,
        )
