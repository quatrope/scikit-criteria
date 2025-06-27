#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implements the Additive Ratio Assessment (Balezentiene & Kusta, 2012)."""


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
# ARAS
# =============================================================================


def aras(matrix, weights, ideal):
    """Execute the ARAS method without any validation.

    This function assumes that the ideal alternative has already been extracted
    from the decision matrix and provided separately.

    The results returned by this function do not include the ideal alternative
    itself.

    Parameters
    ----------
    matrix : ndarray of shape (n_alternatives, n_criteria)
        The decision matrix excluding the ideal alternative.
    weights : ndarray of shape (n_criteria,)
        The weight of each criterion.
    ideal : ndarray of shape (n_criteria,)
        The ideal alternative, provided separately.
    """
    # apply weights to the deision matrix and the ideal alternative
    wmtx = np.multiply(matrix, weights)
    wideal = np.multiply(ideal, weights)

    # compute the weighted score of each alternative
    score = np.sum(wmtx, axis=1)
    ideal_score = np.sum(wideal)

    # calculate utility values
    utility = score / ideal_score

    return (
        rank.rank_values(utility, reverse=True),
        score,
        utility,
        ideal_score,
    )


class ARAS(SKCDecisionMakerABC):
    """Additive Ratio Assessment (ARAS).

    ARAS is a multi-criteria decision-making (MCDM) method that ranks
    alternatives based on their aggregated performance with respect to an
    explicitly provided ideal alternative.

    Each alternative's score is computed by summing its weighted values across
    all criteria. The utility of an alternative is then defined as the ratio of
    its score to the score of the ideal alternative. The higher this utility,
    the better the alternative ranks.

    This implementation requires a user-supplied ideal vector, taken as the
    first row of the decision matrix. All objectives must be of maximization
    type; minimization is not supported and will raise an error.

    Raises
    ------
    ValueError
        If any objective is set to `Objective.MIN`.
    ValueError
        If the ideal is not coherent with the maximization objective
        (i.e., is lower than the observed maximum in the matrix).

    References
    ----------
    :cite:p:`zavadskas2010new`
    """

    _skcriteria_parameters = []

    def _check_ideal(self, matrix, ideal):
        """
        Validate that the ideal vector is coherent with ARAS assumptions.

        This method checks whether each value in the provided `ideal` vector is
        greater than or equal to the maximum value observed in the
        corresponding column (criterion) of the decision matrix.

        ARAS assumes all objectives are to be maximized, so the ideal must
        fulfill the above. This ensures that the utility of each alternative
        (relative to the ideal) is a value in [0, 1].

        Parameters
        ----------
        matrix : array_like
            The decision matrix excluding the ideal alternative.

        ideal : ndarray of shape (n_criteria,)
            The ideal alternative.
        """
        # extract the maxima of each column of the matrix
        maxs = np.max(matrix, axis=0)

        # check if the ideal is greater than or equal to the maxs
        return np.all(maxs <= ideal)

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(
        self, matrix, objectives, weights, **kwargs
    ):
        if Objective.MIN.value in objectives:
            raise ValueError(
                "ARAS can't operate with minimization objectives. "
                "Consider reversing the weights."
            )

        ideal = matrix[0]  # First row is the ideal
        matrix = matrix[1:]  # Remove ideal from decision matrix

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

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        alternatives = alternatives[1:]
        return RankResult(
            "ARAS", alternatives=alternatives, values=values, extra=extra
        )
