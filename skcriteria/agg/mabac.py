#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of Multi-Attributive Border Approximation Area Comparison \
(MABAC) method."""

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
# MABAC
# =============================================================================


def mabac(matrix, weights):
    """Execute MABAC without any validation."""
    # Create normalized matrix maintaining original column order

    weighted_matrix = (matrix + 1) * weights

    # border approximation area (BAA)
    m = matrix.shape[0]  # number of alternatives
    border_approximation_area = np.prod(weighted_matrix, axis=0) ** (1 / m)

    # distance from BAA
    distance = weighted_matrix - border_approximation_area

    score = np.sum(distance, axis=1)

    # ranking (higher score is better)
    return (
        rank.rank_values(score, reverse=True),
        score,
        border_approximation_area,
    )


class MABAC(SKCDecisionMakerABC):
    """Multi-Attributive Border Approximation Area Comparison (MABAC) method.

    MABAC is a multi-criteria decision-making method that determines
    the distance of each alternative from the border approximation
    area. The method is based on the concept of border approximation
    area (BAA), which is calculated as the geometric mean of the
    weighted normalized decision matrix.

    The method consists of the following steps::

        1. Normalization of the decision matrix
        2. Calculation of the weighted normalized decision matrix
        3. Determination of the border approximation area (BAA)
        4. Calculation of the distance from BAA
        5. Calculation of the final score

    References
    ----------
    :cite:p:`pamucar20153016`

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(objectives == Objective.MIN):
            raise ValueError("MABAC does not support minimization objectives.")
        rank, score, border_approximation_area = mabac(matrix, weights)
        return rank, {
            "score": score,
            "border_approximation_area": border_approximation_area,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MABAC",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
