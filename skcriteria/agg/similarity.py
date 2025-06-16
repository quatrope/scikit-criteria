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
    )


class ARAS(SKCDecisionMakerABC):
    """Additive Ratio Assessment (ARAS).

    ARAS (Additive Ratio Assessment) is a multi-criteria decision-making (MCDM)
    method that ranks alternatives based on their aggregated performance with
    respect to an explicitly provided ideal solution.

    Each alternative is evaluated by summing its weighted performance scores
    across all criteria, and then comparing that sum to the ideal score.
    The closer the total score is to the ideal, the better the alternative
    ranks.

    This implementation **requires** a user-supplied ideal vector, taken as the
    first row of the decision matrix. All objectives must be of maximization
    type; minimization is not supported and will raise an error.

    Raises
    ------
    ValueError
        If any objective is set to `Objective.MIN`.
    ValueError
        If the extracted ideal is not coherent with the maximization objective
        (i.e., is lower than the observed maximum in the matrix).

    References
    ----------
    :cite:p:`zavadskas2010new`
    """

    _skcriteria_parameters = []

    def _check_ideal(self, matrix, ideal):
        """
        Validate that the provided ideal vector is coherent with ARAS assumptions.

        This method checks whether each value in the provided `ideal` vector is
        greater than or equal to the maximum value observed in the corresponding
        column (criterion) of the decision matrix.

        ARAS assumes all objectives are to be maximized, so the ideal must dominate
        all alternatives in every criterion. This ensures that the utility of each
        alternative (relative to the ideal) is a value in [0, 1].

        Parameters
        ----------
        matrix : array_like
            The decision matrix containing one row per alternative and one column
            per criterion. Must be at least 2D.
        ideal : array_like
            A 1D array containing the ideal value for each criterion. Must have the
            same number of elements as columns in `matrix`.

        Returns
        -------
        bool
            True if the ideal vector is valid (i.e., all ideal[i] >= max(matrix[:, i])),
            False otherwise.
        """

        # extract the maxima of each column of the matrix
        maxs = np.max(matrix, axis=0)

        # check if the ideal is greater than or equal to the maxs
        return np.all(maxs <= ideal)

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, ideal, **kwargs):
        if Objective.MIN.value in objectives:
            raise ValueError(
                "ARAS can't operate with minimization objectives. "
                "Consider reversing the weights."
            )

        if not self._check_ideal(matrix, ideal):
            raise ValueError(
                "Invalid ideal vector: all ideal values must be greater than"
                "or equal to the maximum observed value for each corresponding"
                "criterion (ARAS assumes maximization objectives only)."
            )

        ranking, scores, utility, ideal_score = aras(matrix, weights, ideal)
        return ranking, {
            "score": scores,
            "utility": utility,
            "ideal_score": ideal_score,
        }

    def _prepare_data(self, **kwargs):
        kwargs["ideal"] = kwargs["matrix"][0]
        kwargs["matrix"] = kwargs["matrix"][1:]
        kwargs["alternatives"] = kwargs["alternatives"][1:]
        return kwargs

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ARAS", alternatives=alternatives, values=values, extra=extra
        )
