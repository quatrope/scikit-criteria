#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
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

from ..core import Objective, RankResult, SKCDecisionMakerABC
from ..utils import doc_inherit, rank

# =============================================================================
# CONSTANTS
# =============================================================================

_VALID_DISTANCES = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "wminkowski",
    "yule",
]


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
    ideal = np.where(objectives == Objective.MAX.value, maxs, mins)
    anti_ideal = np.where(objectives == Objective.MIN.value, maxs, mins)

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
    **kwargs : dict, optional
        Extra arguments to metric: refer to each metric documentation for a
        list of all possible arguments.
        Some possible arguments:

        - p : scalar The p-norm to apply for Minkowski, weighted and
          unweighted. Default: 2.
        - w : array_like The weight vector for metrics that support weights
          (e.g., Minkowski).
        - V : array_like The variance vector for standardized Euclidean.
          Default: var(vstack([XA, XB]), axis=0, ddof=1)
        - VI : array_like The inverse of the covariance matrix for
          Mahalanobis. Default: inv(cov(vstack([XA, XB].T))).T

        This extra parameters are passed to ``scipy.spatial.distance.cdist``
        function,

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

    def __init__(self, *, metric="euclidean", **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    @property
    def metric(self):
        """Which distance metric will be used."""
        return self._metric

    @metric.setter
    def metric(self, metric):
        if not callable(metric) and metric not in _VALID_DISTANCES:
            raise ValueError(f"Invalid metric '{metric}'")
        self._metric = metric

    @property
    def kwargs(self):
        """Extra parameters for the ``scipy.spatial.distance.cdist()``."""
        return self._kwargs

    @kwargs.setter
    def kwargs(self, kwargs):
        self._kwargs = dict(kwargs)

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if Objective.MIN.value in objectives:
            warnings.warn(
                "Although TOPSIS can operate with minimization objectives, "
                "this is not recommended. Consider reversing the weights "
                "for these cases."
            )
        rank, ideal, anti_ideal, similarity = topsis(
            matrix, objectives, weights, metric=self._metric, **self._kwargs
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
