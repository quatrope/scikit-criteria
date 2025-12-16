#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""MAIRCA (Multi Attributive Ideal Real Comparative Analysis) Method."""


# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np
    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank

# =============================================================================
# MAIRCA
# =============================================================================


def mairca(matrix, objectives, weights, P_ai):
    """Execute MAIRCA without any validation."""
    T_p = P_ai[:, np.newaxis] * weights[np.newaxis, :]

    mask = objectives == Objective.MAX.value
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)
    T_r = np.zeros_like(T_p)

    for j in range(matrix.shape[1]):
        if min_vals[j] == max_vals[j]:
            T_r[:, j] = T_p[:, j]
        else:
            if mask[j]:
                T_r[:, j] = T_p[:, j] * (
                    (matrix[:, j] - min_vals[j]) / (max_vals[j] - min_vals[j])
                )
            else:
                T_r[:, j] = T_p[:, j] * (
                    (matrix[:, j] - max_vals[j]) / (min_vals[j] - max_vals[j])
                )

    G = T_p - T_r

    Q_i = G.sum(axis=1)

    ranking = rank.rank_values(Q_i, reverse=False)
    return ranking, Q_i


class MAIRCA(SKCDecisionMakerABC):
    """
    MAIRCA (MultiAtributive Ideal-Real Comparative Analysis).

    MAIRCA was developed in 2014 by the Center for Logistics Research at the
    University of Defence in Belgrade.
    The basic MAIRCA setup is to define the gap observed for each alternative.
    This gap is calculated as the difference between the “theoretical rating
    matrix” and the “real rating matrix.”
    The best alternative is the one with the lowest total gap value.

    Parameters
    ----------
        P_ai : numpy array with len(P_ai) equal to alternatives total.
        Represents preferences for alternatives.

    References
    ----------
    :cite:p:`ljubomir2016combination`
    :cite:p:`aksoy2021analysis`
    :cite:p:`dragan2018hybrid`
    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, P_ai=None, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError("MAIRCA can't operate with values <= 0")
        if P_ai is None:
            m = len(matrix)
            P_ai = np.ones(m) / m
        if len(P_ai) != len(matrix):
            raise ValueError(
                "Length of P_ai must match number of alternatives"
            )
        if not np.isclose(np.sum(P_ai), 1):
            raise ValueError("Sum of P_ai must be 1.")
        if np.any(P_ai < 0):
            raise ValueError("P_ai must be non-negative.")

        rank, q_i = mairca(
            matrix,
            objectives,
            weights,
            P_ai,
        )
        return rank, {
            "values": q_i,
        }

    @doc_inherit(SKCDecisionMakerABC.evaluate)
    def evaluate(self, dm, *, P_ai=None):
        return self._evaluate_dm(dm, P_ai=P_ai)

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MAIRCA",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
