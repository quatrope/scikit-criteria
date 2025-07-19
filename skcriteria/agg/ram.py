#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""RAM (Root Assessment Method) decision-making method."""

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
# RAM
# =============================================================================


def ram(matrix, objectives, weights):
    """Execute RAM without any validation."""
    weighted_matrix = np.multiply(matrix, weights)

    sum_benefit = np.sum(
        weighted_matrix[:, objectives == Objective.MAX.value], axis=1
    )
    sum_cost = np.sum(
        weighted_matrix[:, objectives == Objective.MIN.value], axis=1
    )

    score = np.power(2 + sum_benefit, 1 / (2 + sum_cost))

    return rank.rank_values(score, reverse=True), sum_benefit, sum_cost, score


class RAM(SKCDecisionMakerABC):
    r"""Root Assessment Method (RAM).

    RAM calculates the utility value of each alternative by aggregating their
    scores over the criteria, treating beneficial and non-beneficial criteria
    differently. The method uses an aggregation function that combines
    compensatory and partially compensatory characteristics.

    The aggregation function used is:

    .. math::
        RI_i = \sqrt[2 + S_{-i}]{2 + S_{+i}}

    where :math:`S_{+i}` is the weighted sum of beneficial criteria for
    alternative :math:`i`,
    :math:`S_{-i}` is the weighted sum of non-beneficial criteria for
    alternative :math:`i`.
    :math:`RI_i` is the score of alternative :math:`i` based on the RAM method.
    The ranking is done based on the score, where higher scores indicate better
    alternatives.

    To use this method, the decision matrix should be normalized using the
    following formula:

    .. math::
        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    References
    ----------
    :cite:p:`SOTOUDEHANVARI2023138695`

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        rank, sum_benefit, sum_cost, score = ram(matrix, objectives, weights)
        return rank, {
            "sum_benefit": sum_benefit,
            "sum_cost": sum_cost,
            "score": score,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "RAM",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
