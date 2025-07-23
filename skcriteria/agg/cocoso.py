#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
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
    from ..core import Objective
    from ..utils import doc_inherit, rank


# =============================================================================
# CoCoSo
# =============================================================================


def cocoso(matrix, weights, lambda_value):
    """Execute CoCoSo without any validation."""
    score_wsm = np.sum(matrix * weights, axis=1)
    score_wpm = np.sum(matrix**weights, axis=1)

    # calculate the arithmetic mean of sums of WSM and WPM scores.
    sum_scores = score_wsm + score_wpm
    k_a = sum_scores / np.sum(sum_scores)

    # calculate the sum of relative scores of WSM and WPM compared to the best.
    k_b = score_wsm / np.min(score_wsm) + score_wpm / np.min(score_wpm)

    # calculate the balanced compromise of WSM and WPM models scores.
    k_c = (lambda_value * score_wsm + (1 - lambda_value) * score_wpm) / (
        lambda_value * np.max(score_wsm)
        + (1 - lambda_value) * np.max(score_wpm)
    )

    score = (k_a * k_b * k_c) ** (1 / 3) + (k_a + k_b + k_c) * (1 / 3)

    return rank.rank_values(score, reverse=True), score, k_a, k_b, k_c


class CoCoSo(SKCDecisionMakerABC):
    r"""Combined Compromise Solution (CoCoSo) method.

    The CoCoSo method combines the Weighted Sum Model (WSM) and Weighted
    Product Model (WPM) approaches to provide a comprehensive ranking
    solution for multi-criteria decision-making problems. It uses three
    different aggregation strategies to balance the advantages of both
    WSM and WPM methods.

    The method calculates three compromise scores:

    - **k_a**: Arithmetic mean of normalized WSM and WPM scores
    - **k_b**: Relative scores compared to the best alternatives
    - **k_c**: Balanced compromise using the lambda parameter

    The final score combines these three measures using both geometric and
    arithmetic means to provide a robust ranking.

    Parameters
    ----------
    lambda_value : float, optional (default=0.5)
        Aggregation parameter in [0, 1] that balances WSM and WPM.
        When lambda_value = 0, the method relies more on WPM;
        When lambda_value = 1, the method relies more on WSM;
        When lambda_value = 0.5, both methods have equal influence.

    References
    ----------
    :cite:p:`yazdani2019cocoso`
    """

    _skcriteria_parameters = ["lambda_value"]

    def __init__(self, lambda_value=0.5):
        if not (1 >= lambda_value >= 0):
            raise ValueError(
                f"lambda_value must be a value between 0 and 1. "
                f"Found {lambda_value}"
            )
        self._lambda_value = lambda_value

    @property
    def lambda_value(self):
        """Balance parameter."""
        return self._lambda_value

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if np.any(matrix < 0):
            raise ValueError("CoCoSo can't operate with values <= 0")

        if Objective.MIN.value in objectives:
            raise ValueError("CoCoSo cannot operate on minimising criteria")

        rank, score, k_a, k_b, k_c = cocoso(matrix, weights, self.lambda_value)
        return rank, {"score": score, "k_a": k_a, "k_b": k_b, "k_c": k_c}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "CoCoso", alternatives=alternatives, values=values, extra=extra
        )
