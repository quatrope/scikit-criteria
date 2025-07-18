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
    import warnings

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..utils import doc_inherit, rank

# =============================================================================
# ARAS
# =============================================================================


def aras(matrix, weights, ideal):
    """Execute the ARAS method without any validation.

    This function assumes that the ideal alternative has already been extracted
    from the decision matrix and provided separately.

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
    """Additive Ratio Assessment (ARAS) method.

    ARAS is a multi-criteria decision-making method that ranks alternatives
    based on the ratio of each alternative's weighted score to that of an ideal
    alternative.

    Each alternative's score is computed by summing its weighted values across
    all criteria. The utility of an alternative is then defined as the ratio of
    its score to the score of the ideal alternative. A higher utility value
    indicates a better alternative.

    The ideal alternative is provided through the `ideal` parameter in the
    `evaluate` method, not at instantiation time.

    Warnings
    --------
    UserWarning
        If the provided ideal alternative is not greater than or equal to the
        maximum value for each criterion, some utility scores may exceed 1.

    References
    ----------
    :cite:p:`zavadskas2010new`
    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, ideal, **kwargs):

        ranking, scores, utility, ideal_score = aras(matrix, weights, ideal)

        if np.any(utility > 1):
            invalid_utility_indexes = np.where(utility > 1)[0]
            desc_warn = "\n".join(
                f"  - Index: {idx}, Value of utility: {utility[idx]:.4f}"
                for idx in invalid_utility_indexes
            )
            warnings.warn(
                "Some computed utility are greater than 1, which suggests the "
                "provided ideal alternative may not be greater than or equal "
                "to the maximum values of the decision matrix.\n\n"
                "Details of utilities > 1:\n"
                f"{desc_warn}\n"
                "Please ensure the ideal alternative is valid."
            )

        return ranking, {
            "score": scores,
            "utility": utility,
            "ideal_score": ideal_score,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ARAS", alternatives=alternatives, values=values, extra=extra
        )

    def evaluate(self, dm, *, ideal=None):
        """Evaluate the alternatives in the given decision matrix using ARAS.

        If no ideal is provided when calling `evaluate`, the method
        automatically assumes the ideal as the maximum value in each criterion
        column of the decision matrix and emits a warning.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.
        ideal : ndarray of shape (n_criteria,), optional
            The ideal alternative, if the ideal is None, the method will
            automatically calculate as the maximum value in each criterion.

        Raises
        ------
        ValueError
            If any objective in the decision matrix is set to minimization.
        ValueError
            If the number of criteria in the ideal alternative does not match
            the number in the decision matrix.

        Warnings
        --------
        UserWarning
            If no ideal alternative is provided, the method will use the
            column-wise maxima of the decision matrix as the default ideal.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.
        """
        if np.any(dm.minwhere):
            raise ValueError(
                "ARAS can't operate with minimization objectives. "
                "Consider reversing the weights."
            )
        if ideal is None:
            ideal = np.max(dm.matrix.to_numpy(), axis=0)
        elif dm.matrix.shape[1] != len(ideal):
            raise ValueError(
                "The ideal alternative must have the same number of "
                "criteria as the decision matrix."
            )

        return self._evaluate_dm(dm, ideal=ideal)
