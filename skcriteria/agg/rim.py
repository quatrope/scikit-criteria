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


def _rim_normalize(value, value_range, ref_ideal):
    """
    Normalization function based on the reference range and ideal
    """
    A, B = value_range
    C, D = ref_ideal

    if C <= value <= D:
        return 1.0
    elif A != C and A <= value < C:
        return 1 - min(abs(value - C), abs(value - D)) / abs(A - C)
    elif D != B and D < value <= B:
        return 1 - min(abs(value - C), abs(value - D)) / abs(D - B)
    else:
        raise ValueError(
            "Invalid value to normalize. Outside the accepted range."
        )


def _rim(matrix, weights, ref_ideals, ranges):

    # Normalize the valuation matrix X
    norm_matrix = np.empty_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            norm_matrix[i, j] = _rim_normalize(
                matrix[i, j],
                ranges[j],
                ref_ideals[j],
            )

    # Calculate the weighted normalized matrix
    weighted_matrix = norm_matrix * weights

    # Calculate the variation to the normalized
    # reference ideal for each alternative
    i_plus = np.linalg.norm(
        weighted_matrix - weights, axis=1
    )  # distance to ideal
    i_minus = np.linalg.norm(weighted_matrix, axis=1)  # distance to origin

    # Calculate the relative index of each alternative
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
        RankResult
        """
        data = dm.to_dict()

        if ref_ideals is None or ranges is None:
            raise ValueError("Both `ref_ideals` and `ranges` are required.")

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
            raise ValueError("ranges length must match number of criteria.")

        for i, (ideal, valid_range) in enumerate(zip(ref_ideals, ranges)):
            if not (valid_range[0] <= ideal[0] <= ideal[1] <= valid_range[1]):
                raise ValueError(
                    f"{ideal} must be within ranges[{i}] = {valid_range}"
                )
