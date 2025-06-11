#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""COPRAS (Complex Proportional Assessment) method."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank


def sum_indexes(matrix: np.ndarray, objectives: np.ndarray):
    """Determine the sums of the minimizing and maximizing indexes, respectively."""

    #  Since each column represents a criteria, we must first differenciate those 
    # to be maximised from those to be minimised
    criteria_max = np.compress(Objective.MAX.value == objectives, matrix, axis=1)
    criteria_min = np.compress(Objective.MIN.value == objectives, matrix, axis=1)
    
    # Then, we sum all maximising/minimising values for each alternative solution
    s_max = np.sum(criteria_max, axis=1)
    s_min = np.sum(criteria_min, axis=1)
    
    return s_max, s_min

def determine_significances(s_max, s_min : np.ndarray):
    """Determine the significances of the compared alternatives
    describing the advantages and disadvantages."""

    min_s_min = np.min(s_min)
    
    dividend = min_s_min * np.sum(s_min)
    
    divisor_sum = np.sum(min_s_min / s_min)
    divisor = s_min * divisor_sum
    
    significances = s_max + (dividend / divisor)

    return significances

def copras(matrix, weights, objectives):
    """Execute the COPRAS method without any validation"""
    # Steps
    #   1: Compute the weighted normalised decision-making matrix
    weighted_dm = matrix * weights

    #   2: Calculate the sums of weighted normalised indices describing 
    #      the i^th alternative
    s_max, s_min = sum_indexes(weighted_dm, objectives)

    #   3: Determine the significances of the alternatives describing
    #     their advantages S_+i and disadvantages S_-i
    significances = determine_significances(s_max, s_min)

    #   4: Calculate the utility degree N_i (out of 100) of alternative i
    utility_degrees = (significances / max(significances) * 100)

    #   5: Rank alternatives based on utility
    ranking = rank.rank_values(utility_degrees, reverse=True)
    return ranking, utility_degrees

class COPRAS(SKCDecisionMakerABC):
    r"""The COPRAS method

    The COmplex PRoportional ASsessment (COPRAS) method was introduced by Zavadskas 
    and Kaklauskas and was used to evaluate the superiority of one alternative over another 
    and makes it possible to compare alternatives. It is used to assess the maximizing and 
    minimizing index values, and the effect of maximizing and minimizing indexes of attributes 
    on the results assessment is considered separately.

    Raises
    ------
    ValueError:
        If some value in the matrix is < 0 or if there are no criteria to be minimized.

    References
    ----------
    :cite:p:`zavadskas1996new`
    :cite:p:`organ2016performance`

    """
    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, objectives, **kwargs):
        if np.any(matrix < 0):
            raise ValueError(
                "COPRAS cannot operate with values < 0"
            )
        
        if not (Objective.MIN.value in objectives):
            raise ValueError(
                "COPRAS cannot operate solely on maximising criteria"
            )
        
        ranking, score = copras(matrix, weights, objectives)
        
        return ranking, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "COPRAS",
            alternatives=alternatives,
            values=values,
            extra=extra
        )