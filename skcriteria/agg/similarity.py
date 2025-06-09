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

        def DEBUG(*args):
            import inspect

            frame = inspect.currentframe().f_back
            for arg in args:
                name = [k for k, v in frame.f_locals.items() if v is arg]
                if name:
                    name = name[0]
                else:
                    name = "unknown"
                print(f"{frame.f_lineno} - {name}: {arg}")

        from skcriteria.preprocessing.scalers import (
            matrix_scale_by_cenit_distance as scale,
        )

        # scale maps zenith to 1. We want the opposite, so we invert objectives
        matrix_scaled = scale(matrix, objectives * -1) * weights
        # New criteria: Manhattan distance and Chebyshev distance
        ncriteria_to_2criteria = lambda a: (np.sum(a), np.max(a))
        # N criteria problem -> 2 criteria problem
        distances_matrix = np.apply_along_axis(ncriteria_to_2criteria, 1, matrix_scaled)
        distances_matrix_scaled = scale(distances_matrix, [1, 1])
        # In this new problem weights are [v, 1-v]
        ans = np.dot(distances_matrix_scaled, [self.v, 1 - self.v])

        q_k = ans
        s_k = distances_matrix[:, 0]
        r_k = distances_matrix[:, 1]

        # STEP 4
        # Rank them
        rank_q_k = rank.rank_values(q_k, reverse=False)
        rank_s_k = rank.rank_values(s_k, reverse=False)
        rank_r_k = rank.rank_values(r_k, reverse=False)

        def best(rank):
            return np.where(rank == 1)

        def second_best(rank):
            return np.where(rank == 2)

        # STEP 5
        dq = 1 / (len(matrix) - 1)

        advantage_condition = (
            q_k[second_best(rank_q_k)] - q_k[best(rank_q_k)] >= dq
        )
        aceptable_advantage = np.where(advantage_condition, True, False)

        aceptable_stability = best(rank_q_k) in (
            best(rank_s_k),
            best(rank_r_k),
        )

        empty_array = np.array([0, 0])

        stability_result = np.where(
            aceptable_stability == False,
            [rank_q_k[0], rank_q_k[1]],
            empty_array,
        )
        stability_result = stability_result[stability_result != 0]

        empty_array = np.array([0])

        advantage_result_1 = np.where(
            aceptable_advantage == False, (rank_q_k[0]), empty_array
        )
        advantage_result_1 = advantage_result_1[advantage_result_1 != 0]

        advantage_result_2 = np.where(q_k - q_k[np.where(rank_q_k == 1)] < dq)[
            0
        ]

        compromise_set = np.concatenate(
            (stability_result, advantage_result_1, advantage_result_2)
        )

        extra = {
            # "f_star": zenith,
            # "f_minus": nadir,
            "r_k": r_k,
            "s_k": s_k,
            "q_k": q_k,
            "aceptable_advantage": aceptable_advantage,
            "aceptable_stability": aceptable_stability,
            "compromise_set": compromise_set,
        }

        # return
        return rank_q_k, extra
