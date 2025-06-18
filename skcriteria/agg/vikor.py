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

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..utils import doc_inherit, rank
    from ..preprocessing.scalers import matrix_scale_by_cenit_distance


# =============================================================================
# VIKOR
# =============================================================================


def _scale(matrix, objectives):
    """Scale columns in matrix to [0,1], indicating distance to Zenith.

    See Also
    --------
    skcriteria.preprocessing.scalers.matrix_scale_by_cenit_distance :
        Underlying scaler, without the meaningful warning.

    Warnings
    --------
    UserWarning:
        Division by zero may occur during scaling if any criterion has
        identical values across all alternatives, or there is identical
        group utility or individual regret across all alternatives.
    """
    # matrix_scale_by_cenit_distance maps zenith to 1.
    # We want the opposite so we invert objectives
    objectives = np.asarray(objectives, dtype=float) * -1

    with np.errstate(divide="warn"):
        with warnings.catch_warnings(record=True) as w:
            result = matrix_scale_by_cenit_distance(matrix, objectives)

    if len(w) > 0:
        warnings.warn(
            "Some criteria (or distance) was equal in all alternatives, "
            "so it was ignored. Inspect the Decision Matrix for criteria "
            "with a single value, or the distances r_k, s_k in result.extra_ "
            "to know which."
        )

    return np.nan_to_num(result)


class VIKOR(SKCDecisionMakerABC):
    """The VIKOR Method for Multi-Criteria Decision Making.

    VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje)
    introduces the concept of a compromise solution, which is a feasible
    solution that is closest to the ideal, and represents a balance
    between the majority rule (group utility) and the individual regret
    of the opponent.

    The method evaluates alternatives by converting an n-criteria
    decision problem into a bi-criteria one using the Manhattan distance
    (:math:`S_k`, or group utility) and the Chebyshev distance
    (:math:`R_k`, or individual regret). These are then combined into
    a single aggregated score (:math:`Q_k`) using a weight factor
    :math:`v` that reflects the decision-making strategy: emphasis on
    group utility (high :math:`v`) or individual regret (low :math:`v`).

    VIKOR allows the identification of a single compromise solution if
    the following two conditions are met:

    - Acceptable advantage:
        The best-ranked alternative is sufficiently better than the
        second.
    - Acceptable stability:
        The best-ranked alternative must also be the best in at least
        one of the original distance metrics.

    Otherwise, it identifies a set of compromise solutions.

    Parameters
    ----------
    v : float, optional, default=0.5
        The strategy weight that reflects the decision-making tendency.
        `v = 0` gives full weight to the Chebyshev distance (individual
        regret), `v = 1` gives full weight to the Manhattan distance
        (group utility), and `v = 0.5` balances both.
        Must satisfy `0 <= v <= 1`.

    use_compromise_set : bool, optional, default=True
        If True, all alternatives within the identified compromise set
        are ranked equally at the top position (rank 1).
        If False, only the best :math:`Q_k` remains at the top rank, and
        it is up to the user to examine the compromise set afterwards.

    Warnings
    --------
    UserWarning:
        Division by zero may occur during scaling if any criterion has
        identical values across all alternatives, or there is identical
        group utility or individual regret across all alternatives.

    References
    ----------
    :cite:p:`opricovic2004compromise`
    """

    _skcriteria_parameters = ["v", "use_compromise_set"]

    def __init__(self, *, v=0.5, use_compromise_set=True):
        self._use_compromise_set = bool(use_compromise_set)
        self._v = float(v)
        if not (self._v >= 0 and self._v <= 1):
            raise ValueError(f"'v' must be 0 <= v <= 1. Found {self._v}")

    @property
    def v(self):
        """The strategy weight for VIKOR."""
        return self._v

    @property
    def use_compromise_set(self):
        """Whether to use the compromise set in ranking."""
        return self._use_compromise_set

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            method="VIKOR",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        # (a): Scale the matrix by distance to zenith
        matrix_scaled = _scale(matrix, objectives) * weights

        # (b): Compute Manhattan distance (S) and Chebyshev distance (R)
        distances_matrix = np.column_stack(
            (np.sum(matrix_scaled, axis=1), np.max(matrix_scaled, axis=1))
        )
        # Scaling to minimize distances
        distances_matrix_scaled = _scale(distances_matrix, [-1, -1])

        # (c): Compute Q: weighted sum of our distances with weights [v, 1-v]
        q_k = np.dot(distances_matrix_scaled, [self.v, 1 - self.v])

        # (d): Rank them
        rank_q_k = rank.rank_values(q_k, reverse=False)

        # (e): Check if solution is acceptable
        chosen_qs = np.where(rank_q_k == 1)[0]  # Possibly many qs with rank 1

        best_q_value = q_k[chosen_qs[0]]
        dq = 1 / (len(matrix) - 1)
        qs_with_acceptable_advantage = np.where(q_k - best_q_value < dq)[0]
        has_acceptable_advantage = (
            # chosen_qs always have acc. adv., therefore same len <=> same qs
            len(qs_with_acceptable_advantage)
            == len(chosen_qs)
        )

        # They must also be the best solution of one of the original distances
        bests = np.any(distances_matrix_scaled == 0, axis=1).nonzero()[0]
        has_acceptable_stability = set(chosen_qs).issubset(set(bests))

        if has_acceptable_stability and has_acceptable_advantage:
            # Our solution was good
            compromise_set = chosen_qs
        elif not has_acceptable_stability and has_acceptable_advantage:
            # When unstable, top 2 ranks are chosen
            compromise_set = np.where(rank_q_k <= 2)[0]
        else:
            # If all fails, include all that would have acceptable advantage
            compromise_set = qs_with_acceptable_advantage

        # Reorder ranking so alternatives in compromise set tie at rank 1
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
            "compromise_set": compromise_set,
        }

        return rank_q_k, extra
