#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of a Combined Compromise Solution (CoCoSo) method."""


# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..utils import doc_inherit, rank
    from .simple import wpm, wsm


# =============================================================================
# CoCoSo
# =============================================================================

def cocoso(matrix, objectives, weights, lamdba_value=0.5):
    """Execute COCOSO without any validation."""
    # calculate ranking by inner product
    rank_wsm, score_wsm = wsm(matrix, weights)
    rank_wpm, score_wpm = wpm(matrix, weights)

    # calculate the arithmetic mean of sums of WSM and WPM scores.
    sum_scores = score_wsm + score_wpm
    k_a = sum_scores / np.sum(sum_scores)

    # calculate the sum of relative scores of WSM and WPM compared to the best.
    k_b = score_wsm / np.min(score_wsm) + score_wpm / np.min(score_wpm)
    
    # calculate the balanced compromise of WSM and WPM models scores.
    k_c = lamdba_value * score_wsm + (1 - lamdba_value) * score_wpm / \
        (lamdba_value * np.max(score_wsm) + (1 - lamdba_value) * np.max(score_wpm))
    
    score = (k_a * k_b * k_c) ** (1/3) + (k_a + k_b + k_c) * (1/3)

    return rank.rank_values(score, reverse=True), score

class CoCoSo(SKCDecisionMakerABC):
    r"""Combined Compromise Solution (CoCoSo) method.

    In CoCoSo the suggested approach is based on an integrated simple additive weighting and
    exponentially weighted product model. It can be a compendium of compromise solutions.
    
    Parameters
    ----------
    TODO:
    lambda_value : float, optional (default=0.5)
        Importance to the WSM and WPM results.

    References
    ----------
    :cite:p:
    `Yazdani, Morteza and Zaraté, Pascale and Kazimieras Zavadskas,
    Edmundas and Turskis, Zenonas A Combined Compromise Solution
    (CoCoSo) method for multi-criteria decision-making problems.
    (2019) Management Decision, 57 (9). 2501-2519. ISSN 0025-1747`

    """

    _skcriteria_parameters = ["lambda_value"]

    def __init__(self, lambda_value=0.5):
        if not (1 >= lambda_value >= 0):
            raise ValueError(f"p must be a value between 0 and 1. Found {lambda_value}")
        self._lambda_value = lambda_value

    @property
    def lambda_value(self):
        """lambda_value"""
        return self._lambda_value

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, score = cocoso(matrix, objectives, weights, self._lambda_value)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "CoCoso", alternatives=alternatives, values=values, extra=extra
        )



