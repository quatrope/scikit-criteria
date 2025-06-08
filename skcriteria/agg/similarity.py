#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Methods based on a similarity between alternatives."""

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

# =============================================================================
# ARAS
# =============================================================================

def aras(matrix, weights, ideal):
    """Execute ARAS without any validation"""
    # apply weights
    wmtx = np.multiply(matrix, weights)
    wideal = np.multiply(ideal, weights)

    # calculate optimality function
    score = np.sum(wmtx, axis=1)
    ideal_score = np.sum(wideal)

    # compare variation with the ideal
    utility = score / ideal_score

    return (
        rank.rank_values(utility, reverse=True),
        score,
        utility,
        ideal_score,
        wideal
    )


class ARAS(SKCDecisionMakerABC):
    """Additive Ratio Assessment (ARAS).

    ARAS (Additive Ratio Assessment) is a multi-criteria decision-making method
    based on the principle that the optimal alternative has the greatest
    utility degree compared to the ideal solution. The performance of each
    alternative is calculated as the sum of its weighted criteria values,
    and compared against an aggregated ideal score.

    This implementation allows specifying a custom ideal vector. The ideal
    should be a 1D array containing one ideal value per criterion. If not
    provided (see future support), it should be computed based on the matrix
    and the objective for each criterion.

    Parameters
    ----------
    ideal : array_like
        A 1D array containing the ideal values for each criterion, with the
        same length as the number of columns in the decision matrix. For
        maximization criteria, the ideal should be greater than or equal to
        the maximum observed value. For minimization, it should be less than
        or equal to the minimum observed value.

    Notes
    -----
    Unlike methods based on distance metrics (like TOPSIS), ARAS directly
    compares weighted aggregated values against a reference ideal score.
    This makes it suitable for additive, linear comparisons across criteria.

    The ideal vector is expected to match the dimensionality of the decision
    matrix (i.e., one value per criterion) and to be coherent with the data.

    Warnings
    --------
    UserWarning:
        If some objective is to minimize.

    References
    ----------
    :cite:p:`zavadskas2010new`
    """

    _skcriteria_parameters = ["ideal"]

    def __init__(self, *, ideal): # To do: valor por defecto de ideal
        self._ideal = ideal

    @property
    def ideal(self):
        """Ideal array used to calculate ARAS."""
        return self._ideal

    def _check_ideal(self, matrix, objectives, ideal):
        # Limit values per criterion (max or min depending on the objective)
        bounds = np.where(
            np.equal(objectives, Objective.MAX.value), # To do: max and min como variables distintas
            np.max(matrix, axis=0),
            np.min(matrix, axis=0)
        )

        # Return True if calculated ideal is the first row of the matrix
        return bounds == ideal # To do: ideal no está en la matriz mas

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if Objective.MIN.value in objectives:
            warnings.warn(
                "Although ARAS can operate with minimization objectives, "
                "this is not recommended. Consider reversing the weights "
                "for these cases."
            )

        ranking, scores, utility, ideal_score, wideal = aras(
            matrix,
            weights,
            ideal=self._ideal
        )
        return ranking, {
            "score": scores,
            "utility": utility,
            "ideal_score": ideal_score,
            "weighted ideal": wideal,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ARAS", alternatives=alternatives, values=values, extra=extra
        )