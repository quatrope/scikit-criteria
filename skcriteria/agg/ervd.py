#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of ERVD method."""

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
# ERVD
# =============================================================================


def _increasing_value_function(reference_point, values, alpha, lambd):
    gains = values > reference_point
    losses = ~gains

    result = np.empty_like(values, dtype=float)
    result[gains] = (values[gains] - reference_point) ** alpha
    result[losses] = -lambd * ((reference_point - values[losses]) ** alpha)

    return result


def _decreasing_value_function(reference_point, values, alpha, lambd):
    gains = values < reference_point
    losses = ~gains

    result = np.empty_like(values, dtype=float)
    result[gains] = (reference_point - values[gains]) ** alpha
    result[losses] = -lambd * ((values[losses] - reference_point) ** alpha)

    return result


def ervd(matrix, objectives, weights, reference_points, alpha, lambd):
    """Execute ERVD without any validation."""
    for j in range(matrix.shape[1]):
        if objectives[j] == Objective.MAX.value:
            matrix[:, j] = _increasing_value_function(
                reference_points[j], matrix[:, j], alpha, lambd
            )
        else:
            matrix[:, j] = _decreasing_value_function(
                reference_points[j], matrix[:, j], alpha, lambd
            )

    # create the ideal and the anti ideal arrays
    ideal = np.max(matrix, axis=0)
    anti_ideal = np.min(matrix, axis=0)

    # calculate distances
    s_plus = np.sum(weights * np.abs(matrix - ideal), axis=1)
    s_minus = np.sum(weights * np.abs(matrix - anti_ideal), axis=1)

    # relative closeness
    similarity = s_minus / (s_plus + s_minus)

    return (
        rank.rank_values(similarity, reverse=True),
        similarity,
        ideal,
        anti_ideal,
        s_plus,
        s_minus,
    )


class ERVD(SKCDecisionMakerABC):
    """
    Election based on Relative Value Distances (ERVD) decision-making method.

    This method integrates an s-shape value function, departing from the
    traditional expected utility function, to more accurately capture
    risk-averse and risk-seeking behaviors. ERVD builds upon the foundational
    principles of the TOPSIS method, extending its capabilities by
    incorporating concepts from prospect theory to refine the assessment of
    alternatives based on their relative distances from ideal and anti-ideal
    solutions.

    Parameters
    ----------
    lambd: float, default=2.25
        Represents the attenuation factor of the losse.

    alpha: float, default=0.88
        Diminishing sensitivity parameters.

    References
    ----------
    :cite:p:`shyur2015multiple`
    """

    _skcriteria_parameters = []

    def __init__(self, *, lambd=2.25, alpha=0.88):
        self.lambd = lambd
        self.alpha = alpha

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ERVD", alternatives=alternatives, values=values, extra=extra
        )

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(
        self, matrix, objectives, weights, reference_points, **kwargs
    ):
        rank, similarity, ideal, anti_ideal, s_plus, s_minus = ervd(
            matrix,
            objectives,
            weights,
            reference_points,
            self.alpha,
            self.lambd,
        )

        return rank, {
            "similarity": similarity,
            "ideal": ideal,
            "anti_ideal": anti_ideal,
            "s_plus": s_plus,
            "s_minus": s_minus,
        }

    def _validate_reference_points(self, reference_points, matrix):
        if reference_points is None:
            raise ValueError(
                "Reference points must be provided for ERVD evaluation."
            )
        if len(reference_points) != matrix.shape[1]:
            raise ValueError(
                "Reference points must match the number of criteria in "
                "the decision matrix."
            )

    def evaluate(self, dm, *, reference_points=None):
        """Validate the dm and calculate and evaluate the alternatives.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.
        reference_points: array-like, optional
            Reference points for each criterion.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.
        """
        data = dm.to_dict()

        self._validate_reference_points(reference_points, data["matrix"])

        result_data, extra = self._evaluate_data(
            **data, reference_points=reference_points
        )

        alternatives = data["alternatives"]
        result = self._make_result(
            alternatives=alternatives, values=result_data, extra=extra
        )

        return result
