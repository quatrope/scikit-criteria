#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""COPRAS (Complex Proportional Assessment) method."""

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
# FUNCTIONS
# =============================================================================


def sum_indexes(matrix: np.ndarray, objectives: np.ndarray):
    """
    Determine the sums of the minimizing and maximizing indexes.

    Each column represents a criterion. This function separates those
    to be maximized from those to be minimized, and sums the values
    accordingly for each alternative.
    """
    criteria_max = np.compress(
        objectives == Objective.MAX.value, matrix, axis=1
    )
    criteria_min = np.compress(
        objectives == Objective.MIN.value, matrix, axis=1
    )

    s_max = np.sum(criteria_max, axis=1)
    s_min = np.sum(criteria_min, axis=1)

    return s_max, s_min


def determine_significances(s_max, s_min: np.ndarray):
    """
    Determine the significances of the compared alternatives.

    This reflects the combined advantages and disadvantages of each
    alternative, using the COPRAS method formulas.
    """
    min_s_min = np.min(s_min)

    dividend = min_s_min * np.sum(s_min)
    divisor_sum = np.sum(min_s_min / s_min)
    divisor = s_min * divisor_sum

    significances = s_max + (dividend / divisor)

    return significances


def copras(matrix, weights, objectives):
    """
    Execute the COPRAS method without any validation.

    Steps:
        1. Compute the weighted normalized decision-making matrix.
        2. Calculate sums describing the alternatives.
        3. Determine significances of alternatives.
        4. Calculate the utility degree of each alternative.
        5. Rank the alternatives based on utility.
    """
    weighted_dm = matrix * weights
    s_max, s_min = sum_indexes(weighted_dm, objectives)
    significances = determine_significances(s_max, s_min)
    utility_degrees = significances / max(significances) * 100.0
    ranking = rank.rank_values(utility_degrees, reverse=True)
    return ranking, utility_degrees, significances, s_max, s_min


# =============================================================================
# COPRAS
# =============================================================================


class COPRAS(SKCDecisionMakerABC):
    """
    The COPRAS method.

    The COmplex PRoportional ASsessment (COPRAS) method, introduced by
    Zavadskas and Kaklauskas, is used to evaluate the superiority of
    one alternative over another. It supports comparison of alternatives
    based on maximizing and minimizing index values.

    Raises
    ------
    ValueError
        If any matrix value is < 0, if there are no criteria to minimize
        or if an alternative has all 0s for values in all minimizing
        criteria.

    References
    ----------
    :cite:p:`zavadskas1996new`
    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if np.any(matrix < 0):
            raise ValueError("COPRAS cannot operate with values < 0")

        if not (Objective.MIN.value in objectives):
            raise ValueError(
                "COPRAS cannot operate solely on maximising criteria"
            )

        sum_min = np.sum(
            matrix, axis=1, where=(Objective.MIN.value == objectives)
        )
        if 0 in sum_min:
            raise ValueError(
                "COPRAS cannot operate when an alternative has all 0s"
                "for values in all minimizing criteria"
            )

        ranking, score, significances, s_max, s_min = copras(
            matrix, weights, objectives
        )

        return ranking, {
            "score": score,
            "significances": significances,
            "S_max": s_max,
            "S_min": s_min,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "COPRAS", alternatives=alternatives, values=values, extra=extra
        )
