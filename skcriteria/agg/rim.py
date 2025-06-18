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


def _rim_normalize_matrix(matrix, ref_ideals, ranges):
    norm_matrix = np.asarray(matrix, dtype=float)
    ref_ideals = np.asarray(ref_ideals)
    ranges = np.asarray(ranges)

    A = ranges[:, 0]  # A
    B = ranges[:, 1]  # B
    C = ref_ideals[:, 0]  # C
    D = ref_ideals[:, 1]  # D

    # Condition 1: C <= X <= D → 1.0
    mask1 = (matrix >= C) & (matrix <= D)
    norm_matrix[mask1] = 1.0

    # Condition 2: A <= X < C and A != C
    mask2 = (matrix >= A) & (matrix < C) & (A != C)
    diff_C = np.abs(matrix - C)
    diff_D = np.abs(matrix - D)
    denom = np.abs(A - C)

    new_values = 1 - (np.minimum(diff_C, diff_D) / denom)
    norm_matrix[mask2] = new_values[mask2]

    # Condition 3: D < X <= B and D != B
    mask3 = (matrix > D) & (matrix <= B) & (D != B)
    diff_C = np.abs(matrix - C)
    diff_D = np.abs(matrix - D)
    denom = np.abs(D - B)

    new_values = 1 - (np.minimum(diff_C, diff_D) / denom)
    norm_matrix[mask3] = new_values[mask3]

    return norm_matrix


def _rim(matrix, weights, ref_ideals, ranges):

    norm_matrix = _rim_normalize_matrix(matrix, ref_ideals, ranges)
    weighted_matrix = norm_matrix * weights

    i_plus = np.linalg.norm(weighted_matrix - weights, axis=1)
    i_minus = np.linalg.norm(weighted_matrix, axis=1)

    R = i_minus / (i_plus + i_minus)
    ranking = rank.rank_values(R, reverse=True)

    return ranking, {
        "score": R,
        "norm_matrix": norm_matrix,
        "weighted_matrix": weighted_matrix,
        "i_plus": i_plus,
        "i_minus": i_minus,
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

        extra = {"ref_ideals": ref_ideals, "ranges": ranges, **method_extra}

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
