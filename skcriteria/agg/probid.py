#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Implementation of PROBID and SimplifiedPROBID.

PROBID (Preference Ranking On the Basis of Ideal-Average Distance) and
SimplifiedPROBID (simple variation of PROBID).
"""

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
# BasePROBID
# =============================================================================


class BasePROBID(SKCDecisionMakerABC):
    """
    Base abstract class for PROBID variants.

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
                f"Although {self.__class__.__name__} can operate with "
                "minimization objectives, this is not recommended. Consider "
                "reversing the weights for these cases."
            )
        rank, ideals, pos_ideal, neg_ideal, score = self._method_func(
            matrix,
            objectives,
            weights,
            metric=self.metric,
        )
        return rank, {
            "ideals": ideals,
            "pos_ideal": pos_ideal,
            "neg_ideal": neg_ideal,
            "score": score,
        }


# =============================================================================
# PROBID
# =============================================================================


def probid(matrix, objectives, weights, metric="euclidean", **kwargs):
    """Execute PROBID without any validation."""
    # apply weights
    wmtx = np.multiply(matrix, weights)

    # sort from most PIS to most NIS
    where_max = np.equal(objectives, Objective.MAX.value)
    ideals = np.where(
        where_max,
        np.sort(wmtx, axis=0)[::-1],
        np.sort(wmtx, axis=0),
    )

    # calculate averages
    average = np.mean(wmtx, axis=0)

    # calculate distances
    d_pis = distance.cdist(wmtx, ideals, metric=metric, out=None, **kwargs)
    d_avrg = distance.cdist(
        wmtx, average[True], metric=metric, out=None, **kwargs
    ).T.flatten()

    # calculate the point where the ideal distance is cut
    n_alternatives = len(d_pis)
    median_split = (n_alternatives + (n_alternatives % 2)) // 2

    # calculate the overall positive-ideal distance
    weights = 1 / np.arange(1, median_split + 1)
    pos_ideal = np.sum(d_pis[:, :median_split] * weights, axis=1)

    # calculate the overall negative-ideal distance
    weights = 1 / (
        n_alternatives - np.arange(median_split, n_alternatives + 1) + 1
    )
    start = median_split - 1
    neg_ideal = np.sum(d_pis[:, start:] * weights, axis=1)

    # pos-ideal/neg-ideal ratio
    ratio = pos_ideal / neg_ideal

    # performance score
    score = 1 / (1 + ratio**2) + d_avrg

    # compute the rank and return the result
    return (
        rank.rank_values(score, reverse=True),
        ideals,
        pos_ideal,
        neg_ideal,
        score,
    )


class PROBID(BasePROBID):
    """
    Executes the PROBID method.

    The PROBID method considers a spectrum of ideal solutions and the
    average solution to determine the performance score of each optimal
    solution.
    """

    _method_func = staticmethod(probid)

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "PROBID", alternatives=alternatives, values=values, extra=extra
        )


# =============================================================================
# SimplifiedPROBID
# =============================================================================


def simplifiedprobid(
    matrix, objectives, weights, metric="euclidean", **kwargs
):
    """Execute SimplifiedPROBID without any validation."""
    # apply weights
    wmtx = np.multiply(matrix, weights)

    # sort from most PIS to most NIS
    where_max = np.equal(objectives, Objective.MAX.value)
    ideals = np.where(
        where_max,
        np.sort(wmtx, axis=0)[::-1],
        np.sort(wmtx, axis=0),
    )

    # calculate distances
    d_pis = distance.cdist(wmtx, ideals, metric=metric, out=None, **kwargs)

    # calculate the point where the ideal distance is cut
    n_alternatives = len(d_pis)
    quartile_split = max(1, n_alternatives // 4)

    # calculate the overall positive-ideal distance
    weights = 1 / np.arange(1, quartile_split + 1)
    pos_ideal = np.sum(d_pis[:, :quartile_split] * weights, axis=1)

    # calculate the overall negative-ideal distance
    quartile_split = n_alternatives - quartile_split
    weights = 1 / (n_alternatives - np.arange(quartile_split, n_alternatives))
    neg_ideal = np.sum(d_pis[:, quartile_split:] * weights, axis=1)

    # performance score
    score = neg_ideal / pos_ideal

    # compute the rank and return the result
    return (
        rank.rank_values(score, reverse=True),
        ideals,
        pos_ideal,
        neg_ideal,
        score,
    )


class SimplifiedPROBID(BasePROBID):
    """
    Executes the SimplifiedPROBID method.

    The SimplifiedPROBID method simplifies PROBID method by using only
    the top and bottom quartiles of ideal solutions.
    """

    _method_func = staticmethod(simplifiedprobid)

    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if len(matrix) < 4:
            warnings.warn(
                "SimplifiedPROBID works best with 4 or more alternatives"
                "since it uses quartiles. Consider using PROBID instead"
                "for small datasets."
            )

        return super()._evaluate_data(matrix, objectives, weights, **kwargs)

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "SimplifiedPROBID",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
