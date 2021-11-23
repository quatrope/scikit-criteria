#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of a family of Multi-objective optimization on the basis of \
ratio analysis (MOORA) methods."""


# =============================================================================
# IMPORTS
# =============================================================================
import itertools as it

import numpy as np

from ..core import Objective, RankResult, SKCDecisionMakerABC
from ..utils import doc_inherit, rank

# =============================================================================
# Ratio MOORA
# =============================================================================


def ratio(matrix, objectives, weights):
    """Execute ratio MOORA without any validation."""
    # change the sign the minimization criteria
    # If we multiply by -1 (min) the weights,
    # when we multipliying this weights by the matrix we emulate
    # the -+ ratio mora strategy
    objective_x_weights = weights * objectives

    # calculate ranking by inner prodcut
    rank_mtx = np.inner(matrix, objective_x_weights)
    score = np.squeeze(np.asarray(rank_mtx))
    return rank.rank_values(score, reverse=True), score


class RatioMOORA(SKCDecisionMakerABC):
    r"""Ratio based MOORA method.

    In MOORA the set of ratios are suggested to be normalized as the square
    roots of the sum of squared responses as denominators, but you can
    use any scaler.

    These ratios, as dimensionless, seem to be the best choice among different
    ratios. These dimensionless ratios, situated between zero and one, are
    added in the case of maximization or subtracted in case of minimization:

    .. math::

        Ny_i = \sum_{i=1}^{g} Nx_{ij} - \sum_{i=1}^{g+1} Nx_{ij}

    with:
    :math:`i = 1, 2, ..., g` for the objectives to be maximized,
    :math:`i = g + 1, g + 2, ...,n` for the objectives to be minimized.

    Finally, all alternatives are ranked, according to the obtained ratios.

    References
    ----------
    :cite:p:`brauers2006moora`

    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, score = ratio(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "RatioMOORA", alternatives=alternatives, values=values, extra=extra
        )


# =============================================================================
# Reference Point Moora
# =============================================================================


def refpoint(matrix, objectives, weights):
    """Execute reference point MOORA without any validation."""
    # max and min reference points
    rpmax = np.max(matrix, axis=0)
    rpmin = np.min(matrix, axis=0)

    # merge two reference points acoording objectives
    mask = np.where(objectives == Objective.MAX.value, objectives, 0)
    reference_point = np.where(mask, rpmax, rpmin)

    # create rank matrix
    rank_mtx = np.max(np.abs(weights * (matrix - reference_point)), axis=1)
    score = np.squeeze(np.asarray(rank_mtx))
    return rank.rank_values(score), score, reference_point


class ReferencePointMOORA(SKCDecisionMakerABC):
    r"""Rank the alternatives by distance to a reference point.

    The reference point is selected with the Min-Max Metric of Tchebycheff.

    .. math::

        \min_{j} \{ \max_{i} |r_i - x^*_{ij}| \}

    This reference point theory starts from the already normalized ratios
    as suggested in the MOORA method, namely formula:

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    Preference is given to a reference point possessing as coordinates the
    dominating coordinates per attribute of the candidate alternatives and
    which is designated as the *Maximal Objective Reference Point*. This
    approach is called realistic and non-subjective as the coordinates,
    which are selected for the reference point, are realized in one of the
    candidate alternatives.

    References
    ----------
    :cite:p:`brauers2012robustness`

    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, score, reference_point = refpoint(matrix, objectives, weights)
        return rank, {"score": score, "reference_point": reference_point}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ReferencePointMOORA",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


# =============================================================================
# FULL MULTIPLICATIVE FORM
# =============================================================================


def fmf(matrix, objectives, weights):
    """Execute Full Multiplicative Form without any validation."""
    weighted_matrix = np.log(np.multiply(matrix, weights))

    if Objective.MAX.value in objectives:
        max_columns = weighted_matrix[:, objectives == Objective.MAX.value]
        Aj = np.sum(max_columns, axis=1)
    else:
        Aj = 1.0

    if Objective.MIN.value in objectives:
        min_columns = weighted_matrix[:, objectives == Objective.MIN.value]
        Bj = np.sum(min_columns, axis=1)
    else:
        Bj = 0.0

    score = Aj - Bj

    return rank.rank_values(score, reverse=True), score


class FullMultiplicativeForm(SKCDecisionMakerABC):
    r"""Non-linear, non-additive ranking method method.

    Full Multiplicative Form does not use weights and does not require
    normalization.

    To combine a minimization and maximization of different criteria
    in the same problem all the method uses the formula:

    .. math::

        U'_j = \frac{\prod_{g=1}^{i} x_{gi}}
                   {\prod_{k=i+1}^{n} x_{kj}}

    Where :math:`j` = the number of alternatives;
    :math:`i` = the number of objectives to be maximized;
    :math:`n − i` = the number of objectives to be minimize; and
    :math:`U'_j`: the utility of alternative j with objectives to be maximized
    and objectives to be minimized.

    To avoid underflow, instead the multiplication of the values we add the
    logarithms of the values; so :math:`U'_j`:, is finally defined
    as:

    .. math::

        U'_j = \sum_{g=1}^{i} \log(x_{gi}) - \sum_{k=i+1}^{n} \log(x_{kj})

    Notes
    -----
    The implementation works Instead the multiplication of the values we add
    the logarithms of the values to avoid underflow.

    Raises
    ------
    ValueError:
        If some objective is for minimization or some value in the matrix
        is <= 0.

    References
    ----------
    :cite:p:`brauers2012robustness`

    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError(
                "FullMultiplicativeForm can't operate with values <= 0"
            )
        rank, score = fmf(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "FullMultiplicativeForm",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


# =============================================================================
# MULTIMOORA
# =============================================================================


def multimoora(matrix, objectives, weights):
    """Execute weighted product model without any validation."""
    ratio_rank, ratio_score = ratio(matrix, objectives, weights)
    refpoint_rank, refpoint_score, reference_point = refpoint(
        matrix, objectives, weights
    )
    fmf_rank, fmf_score = fmf(matrix, objectives, weights)

    rank_matrix = np.vstack([ratio_rank, refpoint_rank, fmf_rank]).T

    alternatives = len(matrix)
    score = np.zeros(alternatives)

    # comparamos alternativa a alternativa y vemos cual domina a cual
    # la mas dominante va primero, la menos dominante va ultima
    for idx_a, idx_b in it.combinations(range(alternatives), 2):

        # retrieve the two ranks
        alt_a, alt_b = rank_matrix[[idx_a, idx_b]]

        # calculate the dominance
        dominance = rank.dominance(alt_a, alt_b, reverse=True)

        # if is the same rank we don't increment any alternative
        if dominance.eq == 0:
            dom_idx = idx_a if dominance.aDb > dominance.bDa else idx_b
            score[dom_idx] += 1

    ranking = rank.rank_values(score, reverse=True)

    return (
        ranking,
        score,
        rank_matrix,
        ratio_score,
        refpoint_score,
        fmf_score,
        reference_point,
    )


class MultiMOORA(SKCDecisionMakerABC):
    r"""Combination of RatioMOORA, RefPointMOORA and FullMultiplicativeForm.

    These three methods represent all possible methods with dimensionless
    measures in multi-objective optimization and one can not argue that one
    method is better than or is of more importance than the others; so for
    determining the final ranking the implementation maximizes how many times
    an alternative *i* dominates and alternative *j*.

    Raises
    ------
    ValueError:
        If some objective is for minimization or some value in the matrix
        is <= 0.

    References
    ----------
    :cite:p:`brauers2012robustness`

    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError("MultiMOORA can't operate with values <= 0")
        (
            rank,
            score,
            rank_matrix,
            ratio_score,
            refpoint_score,
            fmf_score,
            reference_point,
        ) = multimoora(matrix, objectives, weights)
        return rank, {
            "score": score,
            "rank_matrix": rank_matrix,
            "ratio_score": ratio_score,
            "refpoint_score": refpoint_score,
            "fmf_score": fmf_score,
            "reference_point": reference_point,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "MultiMOORA",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
