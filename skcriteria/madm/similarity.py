#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Methods based on a similarity between alternatives."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

from scipy.spatial import distance

from ._madm_base import RankResult, SKCDecisionMakerABC
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
