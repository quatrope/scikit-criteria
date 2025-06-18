#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Evaluation based on Distance from Average Solution (EDAS) is introduced for
multi-criteria inventory classification (MCIC) problems. In the method,
we use positive and negative distances from the average solution for appraising
alternatives.

Although the proposed method is used for ABC classification of inventory items,
this method can also be used for MCDM problems. The best alternative in the
proposed method is related to the distance from average solution (AV).

This method is very useful when we have some conflicting criteria.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import warnings

    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, lp, rank

# =============================================================================
# EDAS
# =============================================================================

EPSILON = 1e-10

def distance_from_avg(matrix, objectives, avg):
    pda = np.zeros_like(matrix)
    nda = np.zeros_like(matrix)
    
    is_beneficial = np.array([obj == max for obj in objectives])
    diff_from_avg = matrix - avg

    pda_filtered = np.where(is_beneficial,
                       np.maximum(0, diff_from_avg),
                       np.maximum(0, -diff_from_avg))
    pda = pda_filtered / (avg + EPSILON)

    nda_filtered = np.where(is_beneficial,
                      np.maximum(0, -diff_from_avg),
                      np.maximum(0, diff_from_avg))
    nda = nda_filtered / (avg + EPSILON)

    return pda, nda

def edas(matrix, weights, objectives):
    """Execute edas without any validation"""
    """Step 1: Select criteria"""
    """Step 2: Construct the decision matrix"""

    """Step 3: Determine the average solution for each criteria"""

    average_solution = np.mean(matrix, axis=0)

    """Step 4: Calculate the positive (PDA) and distance (NDA) from average"""
    
    pda , nda = distance_from_avg(matrix, objectives, average_solution)

    """Step 5: Determine the weighted sum of PDA and NDA for all alternatives"""
    
    sum_pda = np.sum(pda * weights, axis=1)
    sum_nda = np.sum(nda * weights, axis=1)

    """Step 6: Normalize the values of weighted sums for all alternatives"""
    
    normalized_sum_pda = sum_pda / (np.max(sum_pda) + EPSILON)
    normalized_sum_nda = 1 - (sum_nda / (np.max(sum_nda) + EPSILON))

    """Step 7: Calculate the appraisal score for all alternatives"""
    
    scores = 0.5 * (normalized_sum_pda + normalized_sum_nda)

    """Step 8: Rank the alternatives according to the decreasing values of the score"""

    return rank.rank_values(scores, reverse=True), scores


class EDAS(SKCDecisionMakerABC):
    r"""EDAS Method

    Evaluation Based on Distance from Average Solution (EDAS) 
    In this method we have two measures dealing with desirability of the
    alternatives. The first measure is the positive distance from average (PDA),
    and the second is the negative distance from average (NDA). These measures
    can show the difference between each alternative and the average solution.
    The evaluation of the alternatives is made according to higher values of PDA
    and lower values of NDA. Higher values of PDA and/or low values of NDA 
    represent that the alternative is better than the average solution. Let's 
    assume we have n alternatives and m criteria

    Raises
    ------
    ValueError:
        If the sum of the weights is other than zero.
        If any of the weights is less than zero or more than 1.
    """
    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if np.sum(weights) != 1:
            raise ValueError("Sum of weights other than zero")
        if np.any(weights) < 0 or np.any(weights) > 1:
            raise ValueError("Weights values must be between 0 and 1")
        rank, score = edas(matrix, weights, objectives)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "EDAS", alternatives=alternatives, values=values, extra=extra
        )
