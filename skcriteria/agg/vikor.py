#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""VIKOR method."""

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
# VIKOR
# =============================================================================


class VIKOR(SKCDecisionMakerABC):
    """The VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje) Method for Multi-Criteria Decision Making.

    VIKOR introduces the concept of a compromise solution, which is a feasible solution that is closest to the ideal,
    and represents a balance between the majority rule (group utility) and the individual regret of the opponent.

    The method evaluates alternatives by converting an n-criteria decision problem into a bi-criteria one using
    the Manhattan distance (S_k) and the Chebyshev distance (R_k). These are then combined into a single aggregated
    score (Q_k) using a weight factor `v` that reflects the decision-making strategy: emphasis on group utility (low `v`)
    or individual regret (high `v`).

    VIKOR allows the identification of a compromise solution if the following two conditions are met:
    - Acceptable advantage: The best-ranked alternative is sufficiently better than the second.
    - Acceptable stability: The best-ranked alternative must also be the best in at least one of the original distance metrics.

    Parameters
    ----------
    v : float, optional, default=0.5
        The strategy weight that reflects the decision-making tendency.
        `v = 0` gives full weight to the Chebyshev distance (individual regret),
        `v = 1` gives full weight to the Manhattan distance (group utility),
        and `v = 0.5` balances both.
        Must satisfy 0 <= v <= 1.

    use_compromise_set : bool, optional, default=True
        If True, all alternatives within the identified compromise set
        are ranked equally at the top position (rank 1).
        If False, only the best Q_k remains at the top rank.

    Warnings
    --------
    UserWarning:
        Division by zero may occur during scaling if any criterion has identical values across all alternatives.

    References
    ----------
    :cite:p:`opricovic2004compromise`
    """
    _skcriteria_parameters = ["v", "use_compromise_set"]

    def __init__(
        self,
        *,
        v=0.5,
        use_compromise_set = True
    ):
        self._v = float(v)
        if not (self._v >= 0 and self._v <= 1):
            raise ValueError(f"'v' must be 0 <= v <= 1. Found {self._v}")
        self._use_compromise_set = use_compromise_set

    @property
    def v(self):
        return self._v
    
    @property
    def use_compromise_set(self):
        return self._use_compromise_set

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

        has_rank1_qs = np.where(
            rank_q_k == 1, 1, 0
        )  # probably can delete this array
        rank1_cnt = np.sum(has_rank1_qs == 1)

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
        chosen_qs = np.where(rank_q_k == 1)  # Possibly many qs with rank 1
        bests = np.any(distances_matrix_scaled == 0, axis=1).nonzero()
        has_acceptable_stability = set(chosen_qs[0]).issubset(set(bests[0]))
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
        if self.use_compromise_set:
            max_compromise_rank = np.max(rank_q_k[compromise_set])
            rank_q_k = np.where(
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
        return rank_q_k, extra
