#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Some simple and compensatory methods."""

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

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if Objective.MIN.value in objectives:
            raise ValueError(
                "WeightedSumModel can't operate with minimize objective"
            )
        if np.any(matrix < 0):
            raise ValueError("WeightedSumModel can't operate with values < 0")

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

    _skcriteria_parameters = []

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


# =============================================================================
# WASPAS
# =============================================================================


def waspas(matrix, weights, lambda_value=0.5):
    """Execute WASPAS without any validation."""
    _, wsm_scores = wsm(matrix, weights)

    _, log10_wpm_scores = wpm(matrix, weights)
    wpm_scores = np.power(10, log10_wpm_scores)

    score = lambda_value * wsm_scores + (1 - lambda_value) * wpm_scores
    ranking = rank.rank_values(score, reverse=True)

    return (
        ranking,
        wsm_scores,
        log10_wpm_scores,
        score,
    )


class WeightedAggregatedSumProductAssessment(SKCDecisionMakerABC):
    r"""The Weighted Aggregated Sum Product ASsessment method.

    WASPAS is a multicriteria decision analysis method that combines the
    Weighted Sum Model (WSM) and the Weighted Product Model (WPM) using
    an aggregation parameter :math:`\lambda \in [0, 1]`.

    It is very important to state here that it is applicable only
    when all the data are expressed in exactly the same unit. If this
    is not the case, then the final result is equivalent to "adding
    apples and oranges". To avoid this problem a previous
    normalization step is necessary.

    In general, suppose that a given MCDA problem is defined on :math:`m`
    alternatives and :math:`n` decision criteria. Let :math:`w_j` denote
    the weight of criterion :math:`C_j`, and :math:`a_{ij}` be the
    performance value of alternative :math:`A_i` with respect to
    criterion :math:`C_j`.

    The WASPAS score of alternative :math:`A_i` is defined as:

    .. math::

        A_i^{WASPAS} = \lambda \cdot \sum_{j=1}^{n} w_j a_{ij} +
                    (1 - \lambda) \cdot \prod_{j=1}^{n} a_{ij}^{w_j}

    By default, :math:`\lambda = 0.5`.

    Raises
    ------
    ValueError:
        If some objective is for minimization,
        or some value in the matrix is <= 0,
        or if the parameter `lambda_value` is not in the range [0, 1].

    References
    ----------
    :cite:p:`zavadskas2012optimization`
    """

    _skcriteria_parameters = ["lambda_value"]

    def __init__(self, lambda_value=0.5):
        lambda_value = float(lambda_value)
        if not (1 >= lambda_value >= 0):
            raise ValueError(
                "WeightedAggregatedSumProductAssessment"
                " requires 'lambda_value' to be between"
                f" 0 and 1, but found {lambda_value}."
            )
        self._lambda_value = lambda_value

    @property
    def lambda_value(self):
        """Aggregation parameter λ ∈ [0, 1] that balances WSM and WPM."""
        return self._lambda_value

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if Objective.MIN.value in objectives:
            raise ValueError(
                "WeightedAggregatedSumProductAssessment can't"
                " operate with minimize objective"
            )
        if np.any(matrix <= 0):

            raise ValueError(
                "WeightedAggregatedSumProductAssessment can't"
                " operate with values <= 0"
            )

        (rank, wsm_scores, log10_wpm_scores, score) = waspas(
            matrix, weights, self._lambda_value
        )
        return rank, {
            "wsm_scores": wsm_scores,
            "log10_wpm_scores": log10_wpm_scores,
            "score": score,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "WeightedAggregatedSumProductAssessment",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


# =============================================================================
# SPOTIS
# =============================================================================


def spotis(matrix, weights, bounds, isp):
    """Execute SPOTIS method"""
    
    min_bounds = bounds[:,0]
    max_bounds = bounds[:,1]

    # Calculate alternatives distances to ISP and normalize it
    normalized_distance = np.abs((matrix - isp) / (max_bounds - min_bounds))
    
    # Scores by weighted sum of normalized distances
    scores = np.sum(normalized_distance * weights, axis=1)
    
    return rank.rank_values(scores), {"score": scores}


class SPOTIS(SKCDecisionMakerABC):
    r"""The Stable Preference Ordering Towards Ideal Solution (SPOTIS) method.

    The SPOTIS method is a multi-criteria decision analysis method that is exempt of rank reversal.
    The method is rank reversal free because the preference ordering established from the score matrix
    of the MCDM problem does not require relative comparisons between the alternatives, but only
    comparisons with respect to the ideal solution chosen by the MCDM designer (ISP).

    Raises
    ------
    ValueError:
        - If the bounds are provided and the matrix has values out of the bounds.
        - If the ISP is provided and the ISP has values out of the bounds (either given or calculated from the matrix).
        - If the bounds or ISP have an invalid shape.

    References
    ----------
    :cite:p:`dezert2020spotis`
    """

    _skcriteria_parameters = ["bounds", "isp"]

    def __init__(self, bounds = None, isp = None):
        self._bounds = bounds
        self._isp = isp

        if bounds is not None and isp is not None:
            self._validate_isp(isp, bounds)


    @property
    def bounds(self):
        """Bounds of the criteria."""
        return self._bounds

    @property
    def isp(self):
        """Ideal Solution Point."""
        return self._isp

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if self._bounds is None:
            self._bounds = self._bounds_from_matrix(matrix)
        
        if self._isp is None:
            self._isp = self._isp_from_bounds(self._bounds, objectives)

        self._validate_bounds(self._bounds, matrix)
        self._validate_isp(self._isp, self._bounds)

        extra = {
            "bounds": self._bounds,
            "isp": self._isp
        }

        rank, method_extra = spotis(matrix, weights, self._bounds, self._isp)
        extra.update(method_extra)

        return rank, extra

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "SPOTIS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
    
    def _bounds_from_matrix(self, matrix):
        """Calculate the bounds of the problem from the matrix."""

        min_bounds = np.min(matrix, axis=0).reshape(-1, 1)
        max_bounds = np.max(matrix, axis=0).reshape(-1, 1)
        return np.hstack((min_bounds, max_bounds))
    
    
    def _isp_from_bounds(self, bounds, objectives):
        """Calculate the reference or nominal Ideal Solution Point from the bounds and objectives."""

        row_indexs = np.arange(bounds.shape[0])
        col_indexs = [0 if obj == Objective.MIN.value else 1 for obj in objectives]
        isp = bounds[row_indexs, col_indexs]

        return isp
    
    def _validate_bounds(self, bounds, matrix):
        if bounds.shape != (matrix.shape[1], 2):
            raise ValueError(f"Invalid shape for bounds. It must be (n_criteria, 2). Got: {bounds.shape}.")
        
        min_bounds, max_bounds = bounds[:,0], bounds[:,1]

        within_bounds = (matrix >= min_bounds) & (matrix <= max_bounds)
        if not np.all(within_bounds):
            raise ValueError("The matrix values must be within the provided bounds.")
        
    def _validate_isp(self, isp, bounds):
        if isp.shape[0] != bounds.shape[0]:
            raise ValueError(f"Invalid shape for Ideal Solution Point (ISP). It must have the same number of criteria as the bounds. Got: {isp.shape}.")
        
        min_bounds, max_bounds = bounds[:,0], bounds[:,1]
        if not np.all(isp >= min_bounds) or not np.all(isp <= max_bounds):
            raise ValueError("The isp values must be within the provided bounds.")
        

