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


def DEBUG(*ass):
    from inspect import currentframe as c

    f = c().f_back
    for a in ass:
        n = [k for k, v in f.f_locals.items() if v is a] + ["?"]
        print(f.f_lineno, n[0], a)


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

    def __init__(
        self,
        *,
        v=0.5,
    ):
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

    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        from skcriteria.preprocessing.scalers import (
            matrix_scale_by_cenit_distance as scale,
        )

        # TODO: Check there are no criteria with only one value, as this will
        #       cause division by zero in the scaling step.

        # scale maps zenith to 1. We want the opposite, so we invert objectives
        matrix_scaled = scale(matrix, objectives * -1) * weights

        # New criteria: Manhattan distance and Chebyshev distance
        def ncriteria_to_2criteria(alternative):
            return (np.sum(alternative), np.max(alternative))

        # N criteria problem -> 2 criteria problem
        distances_matrix = np.apply_along_axis(
            ncriteria_to_2criteria, 1, matrix_scaled
        )
        # distances_matrix = np.column_stack((
        #     np.sum(matrix_scaled), np.max(matrix_scaled))
        # ) # Alternative, to be checked
        distances_matrix_scaled = scale(distances_matrix, [1, 1])
        # We now do weighted sum of our 2 criteria with weights [v, 1-v]
        q_k = np.dot(distances_matrix_scaled, [self.v, 1 - self.v])
        # Rank them
        rank_q_k = rank.rank_values(q_k, reverse=False)

        # Check if solution is acceptable
        # has_rank1_qs = np.where(rank_q_k == 1, 1, 0)  # probably can delete this array
        rank1_cnt = np.sum(rank_q_k == 1)

        # chosen_qs = np.where(rank_q_k == 1)  # Possibly many qs with rank 1
        # best_q_value = q_k[chosen_qs[0][0]]  # The value of the best q
        best_q_value = np.min(q_k)

        # DEBUG(chosen_qs,best_q_value, best_qq_value)
        dq = 1 / (len(matrix) - 1)
        qs_with_acceptable_advantage = np.where(q_k - best_q_value < dq)

        # chosen_qs always have acc. adv., therefore same len <=> same qs
        has_acceptable_advantage = (
            len(qs_with_acceptable_advantage[0]) == rank1_cnt
        )
        # They must also be the best solution of one of the original distances
        # DEBUG(distances_matrix_scaled[:,1] * distances_matrix_scaled[:,0])
        # has_any_best_coordinate = np.where(distances_matrix_scaled[:,1] * distances_matrix_scaled[:,0] == 0, 1, 0)
        # stables_rank1_cnt = np.sum(has_rank1_qs * has_any_best_coordinate == 1)

        # has_acceptable_stability = np.isin(chosen_qs, bests).all()
        aux = (
            rank_q_k
            * distances_matrix_scaled[:, 1]
            * distances_matrix_scaled[:, 0]
        )
        zero_cnt = np.sum(aux == 0)
        aux = (
            distances_matrix_scaled[:, 1] * distances_matrix_scaled[:, 0] - aux
        )
        has_acceptable_stability = np.sum(aux == 0) == zero_cnt
        # TODO: Can we iterate over chosen_qs to check for 0s in r,s?
        if has_acceptable_stability and has_acceptable_advantage:
            # Our solution was good
            compromise_set = np.where(rank_q_k == 1)
        elif not has_acceptable_stability and has_acceptable_advantage:
            # When unstable, top 2 ranks are chosen
            compromise_set = np.where(rank_q_k <= 2)
            # TODO: Check whether to include all ranked 2nd or only one
        else:
            # If all fails, include all that would have acceptable advantage
            compromise_set = qs_with_acceptable_advantage

        # This include all variables in compromise set to rank 1. Maybe can be an option ?
        max_compromise_rank = np.max(rank_q_k[compromise_set])
        res = np.where(
            rank_q_k <= max_compromise_rank,
            1,
            rank_q_k - max_compromise_rank + 1,
        )

        extra = {
            "r_k": distances_matrix[:, 1],
            "s_k": distances_matrix[:, 0],
            "q_k": q_k,
            "acceptable_advantage": bool(has_acceptable_advantage),
            "acceptable_stability": bool(has_acceptable_stability),
            "compromise_set": compromise_set[0],
        }
        # TODO: Compromise set should probably return the names of alternatives
        #       not their indices
        # TODO: Should compromise_set affect rank_q_k?

        # return
        return res, extra
