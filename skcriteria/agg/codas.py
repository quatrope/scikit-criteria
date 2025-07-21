#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Combinative Distance-Based Assessment - CODAS.

The CODAS method evaluates alternatives using two distance metrics. It first
calculates both Euclidean and Taxicab distances from a negative-ideal solution,
which represents the worst performance across all criteria.

The method constructs a relative assessment matrix based on these distances,
where the Euclidean distance serves as the primary measure, and the Taxicab
distance acts as a tiebreaker when alternatives are very similar according to
the first. The final ranking is determined by summing the values in the
relative assessment matrix for each alternative, with higher scores indicating
better performance.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    import warnings

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..utils import doc_inherit, rank
    from ..core import Objective


# =============================================================================
# CODAS
# =============================================================================


def _codas_relative_assessment(euclidian_d, taxicab_d, tau):
    """Aux function to construct the relative assessment matrix."""
    euclidian_i = euclidian_d[:, np.newaxis]
    euclidian_k = euclidian_d[np.newaxis, :]

    taxicab_i = taxicab_d[:, np.newaxis]
    taxicab_k = taxicab_d[np.newaxis, :]

    # Differences between distances
    diff_euclidian = np.subtract(euclidian_i, euclidian_k)
    diff_taxicab = np.subtract(taxicab_i, taxicab_k)

    # Condition psi: |diff_euclidian| >= tau --> 1, else --> 0
    psi = (np.abs(diff_euclidian) >= tau).astype(int)

    # Final relative assessment matrix
    rel_assessment_m = np.add(diff_euclidian, np.multiply(psi, diff_taxicab))

    return rel_assessment_m, psi


def codas(matrix, weights, tau):
    """Execute CODAS without any validation and assuming tau value."""
    # Weight the decision matrix
    matrix = np.multiply(matrix, weights)

    # Determine the negative-ideal solution (anti-ideal)
    neg_sol_arr = np.min(matrix, axis=0)

    # Compute Euclidean and Taxicab (Manhattan) distances
    euclidian_distances = np.sqrt(
        np.sum(np.subtract(matrix, neg_sol_arr) ** 2, axis=1)
    )
    taxicab_distances = np.sum(
        np.abs(np.subtract(matrix, neg_sol_arr)), axis=1
    )

    # Build relative assessment matrix
    rel_assessment_m, psi = _codas_relative_assessment(
        euclidian_distances, taxicab_distances, tau
    )

    # Rank alternatives
    score = np.sum(rel_assessment_m, axis=1)

    return rank.rank_values(score, reverse=True), score, psi


class CODAS(SKCDecisionMakerABC):
    """Rank alternatives using CODAS method.

    COmbinative Distance-based ASsessment (CODAS) is an MCDM method that
    ranks alternatives by comparing how far they are from the worst
    possible solution (anti-ideal), simultaneously using Euclidean distance
    as the primary measure and Taxicab (Manhattan) distance as a tiebreaker
    when alternatives are very similar according to the first.


    Parameters
    ----------
    tau : float, optional (default=0.02)
        tau is the threshold parameter that can be set by the decision-maker.
        Used to construct the relative assessment matrix.

    Raises
    ------
    ValueError:
        If the objectives contain a minimize objective.
        If the decision matrix is not normalized.
    UserWarning:
        If tau is not set between 0.01 and 0.05.

    References
    ----------
    :cite:p:`ghorabaee2016new`

    """

    _skcriteria_parameters = ["tau"]

    def __init__(self, tau=0.02):
        self._tau = tau

    @property
    def tau(self):
        """Which tau value will be used."""
        return self._tau

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if Objective.MIN.value in objectives:
            raise ValueError("CODAS can't operate with minimize objective")
        if np.any((matrix > 1) | (matrix < 0)):
            raise ValueError(
                "DM must be normalized (Suggested to use BenefitCostInverter)"
            )
        if (self.tau < 0.01) | (self.tau > 0.05):
            warnings.warn(
                "It is suggested to set tau at a value between 0.01 and 0.05"
            )
        rank, score, psi = codas(matrix, weights, tau=self.tau)
        return rank, {"score": score, "psi": psi}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "CODAS", alternatives=alternatives, values=values, extra=extra
        )
