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
ARAS aggregation method for multi-criteria decision making.

Implements Additive Ratio Assessment (ARAS) as proposed in Balezentiene and
Kusta (2012).
"""


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

    This function assumes that the user has already separated the ideal
    solution from the decision matrix. Typically, the ideal is extracted as
    the **first row** of the original decision matrix provided by the user.

    The ARAS method calculates the utility of each alternative relative to
    the ideal by summing the weighted criteria and normalizing with respect
    to the ideal score.

    Parameters
    ----------
    matrix : ndarray of shape (n_alternatives, m_criteria)
        The decision matrix without the ideal row. Each row represents
        an alternative and each column a criterion.
    weights : ndarray of shape (n_criteria,)
        The weight of each criterion. Must sum to 1.
    ideal : ndarray of shape (n_criteria,)
        The ideal alternative, extracted as the first row of the original
        decision matrix.
    """
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
        Validate that the ideal vector is coherent with ARAS assumptions.

        This method checks whether each value in the provided `ideal` vector is
        greater than or equal to the maximum value observed in the
        corresponding column (criterion) of the decision matrix.

        ARAS assumes all objectives are to be maximized, so the ideal must
        dominate all alternatives in every criterion. This ensures that the
        utility of each alternative (relative to the ideal) is a
        value in [0, 1].

        Parameters
        ----------
        matrix : array_like
            The decision matrix containing one row per alternative and
            one column per criterion. Must be at least 2D.
        ideal : array_like
            A 1D array containing the ideal value for each criterion.
            Must have the same number of elements as columns in `matrix`.

        Returns
        -------
        bool
            True if all ideal[i] >= max(matrix[:, i]),
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
