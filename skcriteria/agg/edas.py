#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================
""" Evaluatio based on Distance from Average Solution (EDAS) is introduced for 
multi-criteria inventory classification (MCIC) problems. In the proposed method, 
we use positive and negative distances fromthe average solution for appraising 
alternatives.

Although the proposed method is used for ABC classification of inventory items, 
this method can also be used for MCDM problems. The best alternative in the 
proposed method is related to the distance from average solution (AV). 
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
    from ..preprocessing.scalers import scale_by_sum
    from ..utils import doc_inherit, lp, rank

# =============================================================================
# EDAS
# =============================================================================

def edas(matrix, weights, objectives):
    """Execute edas without any validation"""
    """Step 1: Select criteria"""
    """Step 2: Construct the decision matrix"""

    """Step 3: Determine the average solution for each criteria"""
    average_solution = np.mean(matrix, axis=0)
    print("avj", average_solution)
    """Step 4: Calculate the positive (PDA) and distance (NDA) from average"""
    pda = np.zeros_like(matrix)
    nda = np.zeros_like(matrix)

    for j in range(matrix.shape[1]):
        is_beneficial = objectives[j] == max
        avg_j = average_solution[j]
        if is_beneficial:
            pda[:, j] = np.divide(np.maximum(0, (matrix[:, j] - avg_j)), avg_j)
            nda[:, j] = np.divide(np.maximum(0, (avg_j - matrix[:, j])), avg_j)
        else:
            pda[:, j] = np.divide(np.maximum(0, (avg_j - matrix[:, j])), avg_j)
            nda[:, j] = np.divide(np.maximum(0, (matrix[:, j] - avg_j)), avg_j)
    
    """Step 5: Determine the weighted sum of PDA and NDA for all alternatives"""
    sum_pda = np.sum(pda * weights, axis=1)
    sum_nda = np.sum(nda * weights, axis=1)

    """Step 6: Normalize the values of weighted sums for all alternatives"""
    normalized_sum_pda = np.divide(sum_pda, np.max(sum_pda))
    normalized_sum_nda = 1 - np.divide(sum_nda, np.max(sum_nda))

    """Step 7: Calculate the appraisal score for all alternatives"""
    scores = np.multiply(0.5, (normalized_sum_pda + normalized_sum_nda))

    """Step 8: Rank the alternatives according to the decreasing values of the score"""
    ranked = -scores
    
    return rank.rank_values(ranked), scores 

class EDAS(SKCDecisionMakerABC):
    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError(
                "Edas can't operate with values <= 0"
            )
        rank, score = edas(matrix, weights, objectives)
        return rank, {"score": score}
    
    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "EDAS",
            alternatives=alternatives,
            values=values,
            extra=extra
        )