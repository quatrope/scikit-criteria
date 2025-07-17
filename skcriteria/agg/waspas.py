#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""WASPAS method."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank
    from .simple import wsm, wpm

# =============================================================================
# WASPAS
# =============================================================================


def waspas(matrix, weights, lambda_value):
    """Execute WASPAS without any validation."""
    _, wsm_scores = wsm(matrix, weights)

    _, log10_wpm_scores = wpm(matrix, weights)
    wpm_scores = np.power(10, log10_wpm_scores)

    score = lambda_value * wsm_scores + (1 - lambda_value) * wpm_scores
    ranking = rank.rank_values(score, reverse=True)

    return (
        ranking,
        wsm_scores,
        log10_wpm_scores,
        score,
    )


class WASPAS(SKCDecisionMakerABC):
    r"""The Weighted Aggregated Sum Product ASsessment method.

    WASPAS is a multicriteria decision analysis method that combines the
    Weighted Sum Model (WSM) and the Weighted Product Model (WPM) using
    an aggregation parameter :math:`\lambda \in [0, 1]`.

    It is very important to state here that it is applicable only
    when all the data are expressed in exactly the same unit. If this
    is not the case, then the final result is equivalent to "adding
    apples and oranges". To avoid this problem a previous
    normalization step is necessary.

    In general, suppose that a given MCDA problem is defined on :math:`m`
    alternatives and :math:`n` decision criteria. Let :math:`w_j` denote
    the weight of criterion :math:`C_j`, and :math:`a_{ij}` be the
    performance value of alternative :math:`A_i` with respect to
    criterion :math:`C_j`.

    The WASPAS score of alternative :math:`A_i` is defined as:

    .. math::

        A_i^{WASPAS} = \lambda \cdot \sum_{j=1}^{n} w_j a_{ij} +
                    (1 - \lambda) \cdot \prod_{j=1}^{n} a_{ij}^{w_j}

    By default, :math:`\lambda = 0.5`.

    Raises
    ------
    ValueError:
        If some objective is for minimization,
        or some value in the matrix is <= 0,
        or if the parameter `lambda_value` is not in the range [0, 1].

    References
    ----------
    :cite:p:`zavadskas2012optimization`
    """

    _skcriteria_parameters = ["lambda_value"]

    def __init__(self, *, lambda_value=0.5):
        lambda_value = float(lambda_value)
        if not (1 >= lambda_value >= 0):
            raise ValueError(
                "WASPAS requires 'lambda_value' to be "
                f"between 0 and 1, but found {lambda_value}."
            )
        self._lambda_value = lambda_value

    @property
    def lambda_value(self):
        """Aggregation parameter in [0, 1] that balances WSM and WPM."""
        return self._lambda_value

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if Objective.MIN in objectives:
            raise ValueError("WASPAS can't operate with minimize objective")
        if np.any(matrix <= 0):
            raise ValueError("WASPAS can't operate with values <= 0")

        (rank, wsm_scores, log10_wpm_scores, score) = waspas(
            matrix, weights, self.lambda_value
        )
        return rank, {
            "wsm_scores": wsm_scores,
            "log10_wpm_scores": log10_wpm_scores,
            "score": score,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "WASPAS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
