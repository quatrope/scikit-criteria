#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Methods based on similarity function."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings


import numpy as np

from ..base import SKCDecisionMakerABC
from ..data import Objective, RankResult
from ..utils import doc_inherit, rank

# =============================================================================
# SAM
# =============================================================================


def topsis(matrix, objectives, weights):
    """Execute TOPSIS without any validation."""
    # apply weights
    wmtx = np.multiply(matrix, weights)

    # extract mins and maxes
    mins = np.min(wmtx, axis=0)
    maxs = np.max(wmtx, axis=0)

    # create the ideal and the anti ideal arrays
    ideal = np.where(objectives == Objective.MAX.value, maxs, mins)
    anti_ideal = np.where(objectives == Objective.MIN.value, maxs, mins)

    # calculate distances
    d_better = np.sqrt(np.sum(np.power(wmtx - ideal, 2), axis=1))
    d_worst = np.sqrt(np.sum(np.power(wmtx - anti_ideal, 2), axis=1))

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

    Warnings
    --------
    UserWarning:
        If some objective is to minimize.

    References
    ----------
    .. [hwang1981methods] Hwang, C. L., & Yoon, K. (1981). Methods for multiple
       attribute decision making. In Multiple attribute decision making
       (pp. 58-191). Springer, Berlin, Heidelberg.
    .. [enwiki:1034743168] TOPSIS. In Wikipedia, The Free Encyclopedia.
       Retrieved from https://en.wikipedia.org/wiki/TOPSIS
    .. [tzeng2011multiple] Tzeng, G. H., & Huang, J. J. (2011).
       Multiple attribute decision making: methods and applications. CRC press.

    """

    @doc_inherit(SKCDecisionMakerABC._validate_data)
    def _validate_data(self, objectives, **kwargs):
        if Objective.MIN.value in objectives:
            warnings.warn(
                "Although TOPSIS can operate with minimization objectives, "
                "this is not recommended. Consider reversing the weights "
                "for these cases."
            )

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, ideal, anti_ideal, similarity = topsis(
            matrix, objectives, weights
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
