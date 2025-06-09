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
# VIKOR
# =============================================================================


class VIKOR(SKCDecisionMakerABC):
    _skcriteria_parameters = ["v"]

    def __init__(self, *, v=0.5):
        self._v = float(v)
        if not (self._v >= 0 and self._v <= 1):
            raise ValueError(f"'v' must be 0 <= v <= 1. Found {self._v}")

    @property
    def v(self):
        return self._v

    def _make_result(self, alternatives, values, extra):
        return RankResult(
            method="VIKOR",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def _check_stability_condition(self, rank0, rank1):
        r0_fpos = np.argwhere(rank0 == 1).squeeze()
        r1_fpos = np.argwhere(rank1 == 1).squeeze()

        if len(r1_fpos):
            pass

        return r0_fpos == r1_fpos

    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        from skcriteria.preprocessing.scalers import (
            matrix_scale_by_cenit_distance as scale,
        )

        # scale maps zenith to 1. We want the opposite, so we invert objectives
        matrix_scaled = scale(matrix, objectives * -1) * weights

        # New criteria: Manhattan distance and Chebyshev distance
        def ncriteria_to_2criteria(alternative):
            return (np.sum(alternative), np.max(alternative))
        # N criteria problem -> 2 criteria problem
        distances_matrix = np.apply_along_axis(
            ncriteria_to_2criteria, 1, matrix_scaled
        )
        distances_matrix_scaled = scale(distances_matrix, [1, 1])
        # We now do weighted sum of our 2 criteria with weights [v, 1-v]
        q_k = np.dot(distances_matrix_scaled, [self.v, 1 - self.v])

        # Opricovic calls Manhattan S and Chebyshev R
        s_k, r_k = distances_matrix.T
        # Rank them
        rank_q_k = rank.rank_values(q_k, reverse=False)
        rank_s_k = rank.rank_values(s_k, reverse=False)
        rank_r_k = rank.rank_values(r_k, reverse=False)

        # Check if solution is acceptable
        best_q = np.where(rank_q_k == 1) # TODO: There could be many qs ranked 1
        dq = 1 / (len(matrix) - 1)
        qs_with_acceptable_advantage = np.where(q_k - q_k[best_q] < dq)
        # Our chosen solution must be the only one with acceptable advantage
        has_acceptable_advantage = np.count_nonzero(qs_with_acceptable_advantage) == 1
        # It must also be the chosen solution of one of the original distances
        has_acceptable_stability = best_q in (
            np.where(rank_s_k == 1),
            np.where(rank_r_k == 1),
        )

        if has_acceptable_stability and has_acceptable_advantage:
            # Our solution was good
            compromise_set = best_q
        elif not has_acceptable_stability and has_acceptable_advantage:
            # When unstable, top-2 are chosen
            compromise_set = np.where(rank_q_k <= 2)
        else:
            # If all fails, include all that would have acceptable advantage
            compromise_set = qs_with_acceptable_advantage

        extra = {
            "r_k": r_k,
            "s_k": s_k,
            "q_k": q_k,
            "acceptable_advantage": has_acceptable_advantage,
            "acceptable_stability": has_acceptable_stability,
            "compromise_set": compromise_set[0],
        }

        # return
        return rank_q_k, extra
