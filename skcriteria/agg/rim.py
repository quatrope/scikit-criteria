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
    import warnings

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank


# =============================================================================
# RIM
# =============================================================================


def _rim_normalize(value, value_range, ref_ideal):
    """
    Normalize a value based on the ideal reference interval and valid range.

    Based on the paper's normalization direction.
    The normalization returns 1.0 if the value is inside
    the reference interval, and decays toward 0.0 when it
    deviates from the ideal, according to its relative distance
    to the boundaries of the valid range.
    """
    range_min, range_max = value_range
    ideal_min, ideal_max = ref_ideal

    if ideal_min <= value <= ideal_max:
        return 1.0
    elif range_min != ideal_min and range_min <= value < ideal_min:
        return 1 - min(abs(value - ideal_min), abs(value - ideal_max)) / abs(
            range_min - ideal_min
        )
    elif ideal_max != range_max and ideal_max < value <= range_max:
        return 1 - min(abs(value - ideal_min), abs(value - ideal_max)) / abs(
            ideal_max - range_max
        )
    raise ValueError(
        f"Value {value} outside normalization range ({range_min}, {range_max})"
    )


def _rim(matrix, weights, ref_ideals, ranges):

    # Normalize the valuation matrix X
    # using the reference ideal and ranges
    # in the sense of the papers RIM
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
    relative_index = i_minus / (i_plus + i_minus)
    ranking = rank.rank_values(relative_index, reverse=True)

    return ranking, {
        "score": relative_index,
        "norm_matrix": norm_matrix,
        "weighted_matrix": weighted_matrix,
        "i_plus": i_plus,
        "i_minus": i_minus,
    }


class RIM(SKCDecisionMakerABC):
    """
    Reference Ideal Method (RIM) for multi-criteria decision analysis.

    RIM ranks alternatives based on their similarity to a user-defined
    *reference ideal region* instead of relying on classical ideal and
    anti-ideal points. This method considers intervals as ideals, allowing
    more flexible and realistic preference modeling.

    The method normalizes the decision matrix values with respect to
    the ideal intervals and the valid ranges of each criterion. Alternatives
    closer to the ideal intervals receive higher scores.

    Parameters
    ----------
    ref_ideals : list of tuple
        Specifies the ideal reference intervals for each criterion.
        Each tuple should be of the form (ideal_min, ideal_max).
        If not provided, the default ideal value for the criteria depends on
        the desired objectives; if it is to be maximized, the highest value
        within the matrix of that criterion will be set as the ideal value,
        and for the criteria to be minimized, the minimum value will be used
        (which generates intervals of length zero).

    ranges : list of tuple
        List of tuples specifying the min and max bounds of each criterion
        Each tuple should be of the form (range_min, range_max).
        If not provided, they are calculated from the maximum
        and minimum values of the decision matrix per criterion.

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

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "RIM",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def _validate_ranges(self, matrix, ref_ideals, ranges):
        """Validates the consistency and format of ref_ideals and ranges."""
        n_criteria = matrix.shape[1]

        if len(ref_ideals) != n_criteria:
            raise ValueError(
                "ref_ideals length must match number of criteria."
            )
        if len(ranges) != n_criteria:
            raise ValueError("Ranges length must match number of criteria.")

        if not all(
            isinstance(ideal, tuple) and len(ideal) == 2
            for ideal in ref_ideals
        ):
            raise TypeError(
                "Each ref_ideal must be a tuple or list of length 2."
            )

        if not all(isinstance(rng, tuple) and len(rng) == 2 for rng in ranges):
            raise TypeError("Each range must be a tuple or list of length 2.")

        for i, (ideal, valid_range) in enumerate(zip(ref_ideals, ranges)):
            if not (valid_range[0] <= ideal[0] <= ideal[1] <= valid_range[1]):
                raise ValueError(
                    f"{ideal} must be within ranges[{i}] = {valid_range}"
                )

    def evaluate(self, dm, *, ref_ideals=None, ranges=None):
        """Validate the decision matrix and calculate a ranking.

        If `ref_ideals` or `ranges` are not provided, default values will be
        automatically inferred:

        - If `ref_ideals` is None, an interval of length zero is created using
          the column-wise maximum (for MAX objectives) or minimum (for MIN
          objectives) from the decision matrix.

        - If `ranges` is None, the valid range for each criterion is set to
          the minimum and maximum values of the corresponding column.

        Parameters
        ----------
        dm : DecisionMatrix
            Decision matrix to evaluate.
        ref_ideals : array-like of tuple, optional
            Reference ideal intervals (per criterion), where each tuple
            defines (ideal_min, ideal_max). If None, a degenerate interval
            is used based on the objectives.
        ranges : array-like of tuple, optional
            Ranges (min, max) for each criterion. If None, calculated
            from column-wise min and max values.

        Warnings
        --------
        UserWarning
            If `ref_ideals` or `ranges` are not provided, default values are
            inferred from the decision matrix.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.
        """
        df_ranges = dm.matrix.agg(["min", "max"])

        if ref_ideals is None:
            where_max = np.equal(dm.objectives, Objective.MAX)
            ideals = np.where(
                where_max, df_ranges.loc["max"], df_ranges.loc["min"]
            )
            ref_ideals = ref_ideals = list(
                map(tuple, np.stack([ideals, ideals], axis=1))
            )
            warnings.warn(
                "No `ref_ideals` were provided. Using default values based "
                "on the objectives of the decision matrix."
                "For MAX objectives, the column maximum is used; "
                "for MIN objectives, the minimum is used. "
                "This produces reference intervals of length zero."
            )
        if ranges is None:
            ranges = list(map(tuple, df_ranges.T.to_numpy()))
            warnings.warn(
                "No `ranges` were provided. "
                "Using the minimum and maximum values of each column"
                "in the decision matrix as default bounds."
            )

        self._validate_ranges(dm.matrix.to_numpy(), ref_ideals, ranges)

        return self._evaluate_dm(dm, ref_ideals=ref_ideals, ranges=ranges)
