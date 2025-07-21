#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Evaluation based on Distance from Average Solution - EDAS.

The EDAS method evaluates alternatives by comparing them to an average solution
benchmark. It calculates two key metrics: Positive Distance from Average (PDA)
for performance exceeding the average, and Negative Distance from Average (NDA)
for performance below average. These measures capture how each alternative
deviates from the mean performance across all criteria.

The final appraisal combines these deviations through a weighted, normalized
scoring process. After computing weighted sums of PDA and NDA for each
alternative, the method normalizes these values and averages them to produce
a comprehensive evaluation score.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank


# =============================================================================
# EDAS
# =============================================================================


def _any_zeroes(matrix):
    """Aux function to avoid division by zero."""
    result = matrix
    if np.any(result == 0):
        result = np.add(result, 1e-5)
    result = np.where(np.equal(result, 0), -1e-6, result)
    return result


def _distance_from_avg(matrix, objectives, avg):
    """Aux function to calculate PDA and NDA."""
    pda = np.zeros_like(matrix, dtype=float)
    nda = np.zeros_like(matrix, dtype=float)

    # Determine if the objective is beneficial or not
    is_beneficial = np.equal(objectives, Objective.MAX.value)

    # Avoid division by zero
    divisor = _any_zeroes(avg)

    diff_from_avg = np.subtract(matrix, avg)
    max_zero_diff = np.maximum(0, diff_from_avg)
    neg_max_zero_diff = np.maximum(0, np.multiply(-1, diff_from_avg))

    # Calculate PDA
    pda_filtered = np.where(is_beneficial, max_zero_diff, neg_max_zero_diff)
    pda = np.divide(pda_filtered, divisor)

    # Calculate NDA
    nda_filtered = np.where(is_beneficial, neg_max_zero_diff, max_zero_diff)
    nda = np.divide(nda_filtered, divisor)

    return pda, nda


def _normalize_sum_pda_nda(pda, nda):
    """Normalize the sums of PDA and NDA."""
    max_pda = np.max(pda)
    max_nda = np.max(nda)

    # Avoid division by zero
    divisor = max_pda if max_pda != 0 else 1e-5
    result_pda = np.divide(pda, divisor)

    # Avoid division by zero
    divisor = max_nda if max_nda != 0 else 1e-5
    result_nda = np.subtract(1, np.divide(nda, divisor))

    return result_pda, result_nda


def edas(matrix, weights, objectives):
    """Execute EDAS without any validation."""
    # Determine the average solution for each criteria
    average_solution = np.mean(matrix, axis=0)

    # Calculate the positive (PDA) and distance (NDA) from average
    pda, nda = _distance_from_avg(matrix, objectives, average_solution)

    # Determine the weighted sum of PDA and NDA for all alternatives
    sum_pda = np.sum(np.multiply(pda, weights), axis=1)
    sum_nda = np.sum(np.multiply(nda, weights), axis=1)

    # Normalize the values of weighted sums for all alternatives
    normal_sum_pda, normal_sum_nda = _normalize_sum_pda_nda(sum_pda, sum_nda)

    # Calculate the appraisal score for all alternatives
    score = np.multiply(0.5, np.add(normal_sum_pda, normal_sum_nda))

    return rank.rank_values(score, reverse=True), score


class EDAS(SKCDecisionMakerABC):
    """Rank alternatives using EDAS method.

    The Evaluation based on Distance from Average Solution (EDAS) method ranks
    alternatives by comparing their performance to the average solution across
    all criteria. For each alternative, it calculates Positive (PDA) and
    Negative (NDA) distances from average values, which are then weighted,
    normalized, and combined into a final appraisal score.

    References
    ----------
    :cite:p:`keshavarz2015multi`

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        rank, score = edas(matrix, weights, objectives)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "EDAS", alternatives=alternatives, values=values, extra=extra
        )
