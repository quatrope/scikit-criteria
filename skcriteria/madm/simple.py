#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Some simple and compensatory methods."""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ..core import Objective, RankResult, SKCDecisionMakerABC
from ..utils import doc_inherit, rank

# =============================================================================
# SAM
# =============================================================================


def wsm(matrix, weights):
    """Execute weighted sum model without any validation."""
    # calculate ranking by inner prodcut

    rank_mtx = np.inner(matrix, weights)
    score = np.squeeze(rank_mtx)

    return rank.rank_values(score, reverse=True), score


class WeightedSumModel(SKCDecisionMakerABC):
    r"""The weighted sum model.

    WSM is the best known and simplest multi-criteria decision analysis for
    evaluating a number of alternatives in terms of a number of decision
    criteria. It is very important to state here that it is applicable only
    when all the data are expressed in exactly the same unit. If this is not
    the case, then the final result is equivalent to "adding apples and
    oranges". To avoid this problem a previous normalization step is necessary.

    In general, suppose that a given MCDA problem is defined on :math:`m`
    alternatives and :math:`n` decision criteria. Furthermore, let us assume
    that all the criteria are benefit criteria, that is, the higher the values
    are, the better it is. Next suppose that :math:`w_j` denotes the relative
    weight of importance of the criterion :math:`C_j` and :math:`a_{ij}` is
    the performance value of alternative :math:`A_i` when it is evaluated in
    terms of criterion :math:`C_j`. Then, the total (i.e., when all the
    criteria are considered simultaneously) importance of alternative
    :math:`A_i`, denoted as :math:`A_{i}^{WSM-score}`, is defined as follows:

    .. math::

        A_{i}^{WSM-score} = \sum_{j=1}^{n} w_j a_{ij},\ for\ i = 1,2,3,...,m

    For the maximization case, the best alternative is the one that yields
    the maximum total performance value.

    Raises
    ------
    ValueError:
        If some objective is for minimization.

    References
    ----------
    :cite:p:`fishburn1967letter`, :cite:p:`enwiki:1033561221`,
    :cite:p:`tzeng2011multiple`

    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if Objective.MIN.value in objectives:
            raise ValueError(
                "WeightedSumModel can't operate with minimize objective"
            )

        rank, score = wsm(matrix, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):

        return RankResult(
            "WeightedSumModel",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


# =============================================================================
# WPROD
# =============================================================================


def wpm(matrix, weights):
    """Execute weighted product model without any validation."""
    # instead of multiply we sum the logarithms
    lmtx = np.log10(matrix)

    # add the weights to the mtx
    rank_mtx = np.multiply(lmtx, weights)

    score = np.sum(rank_mtx, axis=1)

    return rank.rank_values(score, reverse=True), score


class WeightedProductModel(SKCDecisionMakerABC):
    r"""The weighted product model.

    WPM is a popular multi-criteria decision
    analysis method. It is similar to the weighted sum model.
    The main difference is that instead of addition in the main mathematical
    operation now there is multiplication.

    In general, suppose that a given MCDA problem is defined on :math:`m`
    alternatives and :math:`n` decision criteria. Furthermore, let us assume
    that all the criteria are benefit criteria, that is, the higher the values
    are, the better it is. Next suppose that :math:`w_j` denotes the relative
    weight of importance of the criterion :math:`C_j` and :math:`a_{ij}` is
    the performance value of alternative :math:`A_i` when it is evaluated in
    terms of criterion :math:`C_j`. Then, the total (i.e., when all the
    criteria are considered simultaneously) importance of alternative
    :math:`A_i`, denoted as :math:`A_{i}^{WPM-score}`, is defined as follows:

    .. math::

        A_{i}^{WPM-score} = \prod_{j=1}^{n} a_{ij}^{w_j},\ for\ i = 1,2,3,...,m

    To avoid underflow, instead the multiplication of the values we add the
    logarithms of the values; so :math:`A_{i}^{WPM-score}`,
    is finally defined as:

    .. math::

        A_{i}^{WPM-score} = \sum_{j=1}^{n} w_j \log(a_{ij}),\
                            for\ i = 1,2,3,...,m

    For the maximization case, the best alternative is the one that yields
    the maximum total performance value.

    Raises
    ------
    ValueError:
        If some objective is for minimization or some value in the matrix
        is <= 0.

    References
    ----------
    :cite:p:`bridgman1922dimensional`
    :cite:p:`miller1963executive`


    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if Objective.MIN.value in objectives:
            raise ValueError(
                "WeightedProductModel can't operate with minimize objective"
            )
        if np.any(matrix <= 0):
            raise ValueError(
                "WeightedProductModel can't operate with values <= 0"
            )

        rank, score = wpm(matrix, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "WeightedProductModel",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
