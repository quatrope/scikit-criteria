#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""TODO"""

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
# INPUT/OUTPUT PERFORMANCE
# =============================================================================

def _agg_input_performance(in_matrix, in_weights):
    """Calculate aggregate performance for input/non-beneficial/min criteria"""
    # extract mins and maxes of criteria
    mins = np.min(in_matrix, axis=0)
    maxs = np.max(in_matrix, axis=0)

    # performance equation
    perf_in_matrix = ((maxs - in_matrix) * in_weights) / mins

    # aggregated performance
    perf_in = np.sum(perf_in_matrix, axis=1)

    # subtract minimum performance to scale
    perf_in -= np.min(perf_in)
    return perf_in


def _agg_output_performance(out_matrix, out_weights):
    """Calculate aggregate performance for output/beneficial/max criteria"""
    # extract mins of criteria
    mins = np.min(out_matrix, axis=0)

    # performance equation
    perf_out_matrix = ((out_matrix - mins) * out_weights) / mins

    # aggregated performance
    perf_out = np.sum(perf_out_matrix, axis=1)

    # subtract minimum performance to scale
    perf_out -= np.min(perf_out)
    return perf_out


def ocra_performance(matrix, objectives, weights):
    """Compute the overall performance of each alternative by separating criteria"""
    # masks for min and max (non-beneficial and beneficial) objectives
    in_objectives = objectives == Objective.MIN.value
    out_objectives = objectives == Objective.MAX.value

    # compute min and max (in and out) performances, respectively
    perf_in = _agg_input_performance(matrix[:, in_objectives], weights[in_objectives])
    perf_out = _agg_output_performance(matrix[:, out_objectives], weights[out_objectives])

    # combine results and scale
    perf = perf_in + perf_out
    perf -= np.min(perf)

    # compute the rank and return all results
    ranking = rank.rank_values(perf, reverse=True)
    return ranking, perf, perf_in, perf_out


# =============================================================================
# OCRA
# =============================================================================


class OCRA(SKCDecisionMakerABC):
    """"Main OCRA functionality"""

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, perf, perf_in, perf_out = ocra_performance(matrix, objectives, weights)
        return rank, {
            "performance": perf,
            "input_performance": perf_in,
            "output_performance": perf_out,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult("OCRA", alternatives=alternatives, values=values, extra=extra)
    