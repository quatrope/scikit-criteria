#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""RIM reference ideal method."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..utils import doc_inherit, rank


# =============================================================================
# RIM
# =============================================================================


def _rim_normalize_matrix(matrix, ref_intervals, bounds):
    norm_matrix = np.asarray(matrix, dtype=float)
    ref_intervals = np.asarray(ref_intervals)
    bounds = np.asarray(bounds)

    min_vals = bounds[:, 0]  # A: Lower bound
    max_vals = bounds[:, 1]  # B: Upper bound
    ideal_min = ref_intervals[:, 0]  # C: Ideal lower bound
    ideal_max = ref_intervals[:, 1]  # D: Ideal upper bound

    # Condition 1: C <= X <= D → 1.0
    within_ideal = (matrix >= ideal_min) & (matrix <= ideal_max)
    norm_matrix[within_ideal] = 1.0

    # Condition 2: A <= X < C and A != C
    left_side = (
        (matrix >= min_vals) & (matrix < ideal_min) & (min_vals != ideal_min)
    )
    dist_to_ideal_min = np.abs(matrix - ideal_min)
    dist_to_ideal_max = np.abs(matrix - ideal_max)
    denom_left = np.abs(min_vals - ideal_min)

    left_values = 1 - (
        np.minimum(dist_to_ideal_min, dist_to_ideal_max) / denom_left
    )
    norm_matrix[left_side] = left_values[left_side]

    # Condition 3: D < X <= B and D != B
    right_side = (
        (matrix > ideal_max) & (matrix <= max_vals) & (ideal_max != max_vals)
    )
    dist_to_ideal_min = np.abs(matrix - ideal_min)
    dist_to_ideal_max = np.abs(matrix - ideal_max)
    denom_right = np.abs(ideal_max - max_vals)

    right_values = 1 - (
        np.minimum(dist_to_ideal_min, dist_to_ideal_max) / denom_right
    )
    norm_matrix[right_side] = right_values[right_side]

    return norm_matrix


def _rim(matrix, weights, ref_intervals, bounds):

    norm_matrix = _rim_normalize_matrix(matrix, ref_intervals, bounds)
    weighted_matrix = norm_matrix * weights

    distance_to_ideal = np.linalg.norm(weighted_matrix - weights, axis=1)
    distance_to_origin = np.linalg.norm(weighted_matrix, axis=1)

    similarity_ratio = distance_to_origin / (
        distance_to_ideal + distance_to_origin
    )
    ranking = rank.rank_values(similarity_ratio, reverse=True)

    return ranking, {
        "score": similarity_ratio,
        "norm_matrix": norm_matrix,
        "weighted_matrix": weighted_matrix,
        "i_plus": distance_to_ideal,
        "i_minus": distance_to_origin,
    }


class RIM(SKCDecisionMakerABC):
    """Reference Ideal Method (RIM).

    RIM ranks alternatives based on their similarity
    to a user-defined reference ideal region, rather
    than the classical ideal/anti-ideal approach.

    Parameters
    ----------
    ref_ideals : list of tuple
        List of tuples specifying the ideal reference intervals for each
        criterion (e.g., [(30, 35), (0, 0), (100, 120)]).

    ranges : list of tuple
        List of tuples specifying the min and max bounds of each criterion
        (e.g., [(23, 60), (0, 15), (80, 130)]).

    References
    ----------
    :cite:p:`cables2016rim`

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, ref_ideals, ranges, **kwargs):

        ranking, method_extra = _rim(matrix, weights, ref_ideals, ranges)

        extra = {
            "ref_ideals": ref_ideals,
            "ranges": ranges,
            **method_extra,
        }

        return ranking, extra

    def evaluate(self, dm, *, ref_ideals=None, ranges=None):
        """Validate the decision matrix and calculate a ranking.

        Parameters
        ----------
        dm : DecisionMatrix
            Decision matrix to evaluate.
        ref_ideals : array-like
            Reference ideal intervals (per criterion).
        ranges : array-like
            Ranges (min, max) for each criterion.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.
        """
        data = dm.to_dict()

        if ref_ideals is None or ranges is None:
            raise ValueError("Both `ref_ideals` and `ranges` are required.")

        ref_ideals = np.asarray(ref_ideals)
        ranges = np.asarray(ranges)

        self._validate_ranges(data["matrix"], ref_ideals, ranges)

        result_data, extra = self._evaluate_data(
            **data,
            ref_ideals=ref_ideals,
            ranges=ranges,
        )

        return self._make_result(
            alternatives=data["alternatives"],
            values=result_data,
            extra=extra,
        )

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "RIM",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def _validate_ranges(self, matrix, ref_ideals, ranges):
        n_criteria = matrix.shape[1]

        if len(ref_ideals) != n_criteria:
            raise ValueError(
                "ref_ideals length must match number of criteria."
            )
        if len(ranges) != n_criteria:
            raise ValueError("Ranges length must match number of criteria.")

        if ranges.shape != (matrix.shape[1], 2):
            raise ValueError(
                f"Invalid shape for ranges. It must be (n_criteria, 2). \
                Got: {ranges.shape}."
            )

        min_range, max_range = ranges[:, 0], ranges[:, 1]

        ideals_within_ranges = (ref_ideals.T >= min_range) & (
            ref_ideals.T <= max_range
        )
        if not np.all(ideals_within_ranges):
            raise ValueError("Ideals must be within ranges")

        values_within_ranges = (matrix >= min_range) & (matrix <= max_range)
        if not np.all(values_within_ranges):
            raise ValueError(
                "Some values are outside the accepted normalization range."
            )
