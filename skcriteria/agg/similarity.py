#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Methods based on a similarity between alternatives."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import warnings

    import numpy as np

    from scipy.spatial import distance

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank


# =============================================================================
# TOPSIS
# =============================================================================


def topsis(matrix, objectives, weights, metric="euclidean", **kwargs):
    """Execute TOPSIS without any validation."""
    # apply weights
    wmtx = np.multiply(matrix, weights)

    # extract mins and maxes
    mins = np.min(wmtx, axis=0)
    maxs = np.max(wmtx, axis=0)

    # create the ideal and the anti ideal arrays
    where_max = np.equal(objectives, Objective.MAX.value)

    ideal = np.where(where_max, maxs, mins)
    anti_ideal = np.where(where_max, mins, maxs)

    # calculate distances
    d_better = distance.cdist(
        wmtx, ideal[True], metric=metric, out=None, **kwargs
    ).flatten()
    d_worst = distance.cdist(
        wmtx, anti_ideal[True], metric=metric, out=None, **kwargs
    ).flatten()

    # relative closeness
    similarity = d_worst / (d_better + d_worst)

    # compute the rank and return the result
    return (
        rank.rank_values(similarity, reverse=True),
        ideal,
        anti_ideal,
        similarity,
    )


class TOPSIS(SKCDecisionMakerABC):
    """The Technique for Order of Preference by Similarity to Ideal Solution.

    TOPSIS is based on the concept that the chosen alternative should have
    the shortest geometric distance from the ideal solution and the longest
    euclidean distance from the worst solution.

    An assumption of TOPSIS is that the criteria are monotonically increasing
    or decreasing, and also allow trade-offs between criteria, where a poor
    result in one criterion can be negated by a good result in another
    criterion.

    Parameters
    ----------
    metric : str or callable, optional
        The distance metric to use. If a string, the distance function
        can be ``braycurtis``, ``canberra``, ``chebyshev``, ``cityblock``,
        ``correlation``, ``cosine``, ``dice``, ``euclidean``, ``hamming``,
        ``jaccard``, ``jensenshannon``, ``kulsinski``, ``mahalanobis``,
        ``matching``, ``minkowski``, ``rogerstanimoto``, ``russellrao``,
        ``seuclidean``, ``sokalmichener``, ``sokalsneath``,
        ``sqeuclidean``, ``wminkowski``, ``yule``.

    Warnings
    --------
    UserWarning:
        If some objective is to minimize.


    References
    ----------
    :cite:p:`hwang1981methods`
    :cite:p:`enwiki:1034743168`
    :cite:p:`tzeng2011multiple`

    """

    _skcriteria_parameters = ["metric"]

    def __init__(self, *, metric="euclidean"):
        if not callable(metric) and metric not in distance._METRICS_NAMES:
            metrics = ", ".join(f"'{m}'" for m in distance._METRICS_NAMES)
            raise ValueError(
                f"Invalid metric '{metric}'. Plese choose from: {metrics}"
            )
        self._metric = metric

    @property
    def metric(self):
        """Which distance metric will be used."""
        return self._metric

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if Objective.MIN.value in objectives:
            warnings.warn(
                "Although TOPSIS can operate with minimization objectives, "
                "this is not recommended. Consider reversing the weights "
                "for these cases."
            )
        rank, ideal, anti_ideal, similarity = topsis(
            matrix,
            objectives,
            weights,
            metric=self.metric,
        )
        return rank, {
            "ideal": ideal,
            "anti_ideal": anti_ideal,
            "similarity": similarity,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "TOPSIS", alternatives=alternatives, values=values, extra=extra
        )


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

    # Calculate the variation to the normalized reference ideal for each alternative
    i_plus = np.linalg.norm(
        weighted_matrix - weights, axis=1
    )  # distance to ideal
    i_minus = np.linalg.norm(weighted_matrix, axis=1)  # distance to origin

    # Calculate the relative index of each alternative
    R = i_minus / (i_plus + i_minus)

    return R, norm_matrix, weighted_matrix, i_plus, i_minus


class RIM(SKCDecisionMakerABC):
    """Reference Ideal Method (RIM).

    RIM ranks alternatives based on their similarity to a user-defined
    reference ideal region, rather than the classical ideal/anti-ideal approach.

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
        score, norm_matrix, weighted_matrix, i_plus, i_minus = _rim(
            matrix,
            weights,
            ref_ideals,
            ranges,
        )
        return score, {
            "score": score,
            "norm_matrix": norm_matrix,
            "weighted_matrix": weighted_matrix,
            "i_plus": i_plus,
            "i_minus": i_minus,
            "ref_ideals": ref_ideals,
            "ranges": ranges,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "RIM",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
