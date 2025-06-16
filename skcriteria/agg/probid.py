#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""PROBID (Preference Ranking On the Basis of Ideal-Average Distance) method."""

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
# PROBID
# =============================================================================


def probid(matrix, objectives, weights, metric="euclidean", **kwargs):
    """Execute PROBID without any validation."""
    # apply weights
    wmtx = np.multiply(matrix, weights)

    # sort from most PIS to most NIS
    where_max = np.equal(objectives, Objective.MAX.value)

    ideals = np.where(where_max, np.sort(wmtx, axis=0)[::-1], np.sort(wmtx, axis=0))

    # calculate averages
    average = np.mean(wmtx, axis=0)

    # calculate distances
    d_nis = distance.cdist(
        wmtx, ideals, metric=metric, out=None, **kwargs
    )
    d_avrg = distance.cdist(
        wmtx, average[True], metric=metric, out=None, **kwargs
    ).T.flatten()

    # calculate the overall positive-ideal distance
    midpoint = (len(d_nis) + (len(d_nis) % 2)) // 2

    weights = 1 / np.arange(1, midpoint + 1)
    pos_ideal = np.sum(d_nis[:,:midpoint] * weights, axis=1)

    # calculate the overall negative-ideal distance
    weights = 1 / (len(d_nis) - np.arange(midpoint, len(d_nis)+1) + 1)
    neg_ideal = np.sum(d_nis[:,midpoint - 1:] * weights, axis=1)

    # pos-ideal/neg-ideal ratio
    pi_ni_ratio = pos_ideal / neg_ideal

    # performance score
    performance_score = 1 / ( 1 + pi_ni_ratio**2) + d_avrg

    # compute the rank and return the result
    return (
        rank.rank_values(performance_score, reverse=True),
        ideals,
        pos_ideal,
        neg_ideal,
        performance_score,
    )


class PROBID(SKCDecisionMakerABC):
    """The Technique for Order of Preference by Similarity to Ideal Solution.

    The PROBID method considers a spectrum of ideal solutions and the average
    solution to determine the performance score of each optimal solution.

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
    :cite:p:`wang2021preference`

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
                "Although PROBID can operate with minimization objectives, "
                "this is not recommended. Consider reversing the weights "
                "for these cases."
            )
        rank, ideals, pos_ideal, neg_ideal, performance_score = probid(
            matrix,
            objectives,
            weights,
            metric=self.metric,
        )
        return rank, {
            "ideals": ideals,
            "pos_ideal": pos_ideal,
            "neg_ideal": neg_ideal,
            "performance_score": performance_score,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "PROBID", alternatives=alternatives, values=values, extra=extra
        )
