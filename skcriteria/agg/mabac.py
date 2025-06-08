#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of Multi-Attributive Border Approximation Area Comparison (MABAC) method."""

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
# MABAC
# =============================================================================


def mabac(matrix, objectives, weights):
    """Execute MABAC without any validation."""

    # benefit and cost matrices
    benefit_matrix = matrix[:, objectives == Objective.MAX.value]
    cost_matrix = matrix[:, objectives == Objective.MIN.value]

    # max and min for benefit criteria
    max_benefit = np.max(benefit_matrix, axis=0)
    min_benefit = np.min(benefit_matrix, axis=0)

    # normalized benefit matrix
    normalized_benefit = (benefit_matrix - min_benefit) / (max_benefit - min_benefit)

    # max and min for cost criteria
    max_cost = np.max(cost_matrix, axis=0)
    min_cost = np.min(cost_matrix, axis=0)

    # normalized cost matrix
    normalized_cost = (cost_matrix - max_cost) / (min_cost - max_cost)

    # combining normalized matrices
    normalized_matrix = np.concatenate([normalized_benefit, normalized_cost], axis=1)
    
    # weighted normalized decision matrix
    # inverse order of weights
    weights_inverse = weights[::-1]
    weighted_matrix = (normalized_matrix+1) * weights_inverse
    
    # border approximation area (BAA)
    border_approximation_area = np.prod(weighted_matrix, axis=0) ** (1/len(matrix))
    
    # distance from BAA
    distance = weighted_matrix - border_approximation_area
    
    # final score
    score = np.sum(distance, axis=1)
    
    # ranking (higher score is better)
    return rank.rank_values(score, reverse=True), score, border_approximation_area


class MABAC(SKCDecisionMakerABC):
    """Multi-Attributive Border Approximation Area Comparison (MABAC) method.
    
    MABAC is a multi-criteria decision-making method that determines the distance
    of each alternative from the border approximation area. The method is based
    on the concept of border approximation area (BAA), which is calculated as
    the geometric mean of the weighted normalized decision matrix.
    
    The method consists of the following steps:
    1. Normalization of the decision matrix
    2. Calculation of the weighted normalized decision matrix
    3. Determination of the border approximation area (BAA)
    4. Calculation of the distance from BAA
    5. Calculation of the final score
    
    Parameters
    ----------
    matrix : ndarray
        Decision matrix where each row is an alternative and each column is a criterion.
    objectives : ndarray
        Array indicating if each criterion is to be maximized (1) or minimized (-1).
    weights : ndarray
        Array of weights for each criterion.

    Returns
    -------
    tuple
        A tuple containing:
        - rank : ndarray
            Ranking of the alternatives according to MABAC (1 is best)
        - score : ndarray
            Score for each alternative
        - border_approximation_area : ndarray
            Border approximation area values 

    References
    ----------
    :cite:p:`pamucar2015`
    
    """
    
    _skcriteria_parameters = []
    
    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, score, border_approximation_area = mabac(matrix, objectives, weights)
        return rank, {
            "score": score,
            "border_approximation_area": border_approximation_area
        }
    
    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MABAC",
            alternatives=alternatives,
            values=values,
            extra=extra
        ) 