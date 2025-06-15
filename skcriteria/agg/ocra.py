#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of OCRA (Operational Competitiveness Rating) method for \
general MCDM purposes."""

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
    """Calculate aggregate performance for\
    input/non-beneficial/min criteria."""
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
    """Calculate aggregate performance for output/beneficial/max criteria."""
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
    """Compute the overall performance of each alternative."""
    # masks for min and max (non-beneficial and beneficial) objectives
    in_objectives = objectives == Objective.MIN.value
    out_objectives = objectives == Objective.MAX.value

    # compute min and max (in and out) performances, respectively
    perf_in = _agg_input_performance(
        matrix[:, in_objectives],
        weights[in_objectives],
    )
    perf_out = _agg_output_performance(
        matrix[:, out_objectives],
        weights[out_objectives],
    )

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
    r"""OCRA (Operational Competitiveness Rating) method.

    OCRA was initially intended (Parkan, 1994) to maximize the efficiency of
    a Production Unit (PU), seen as a set of activities that consume resources
    (inputs) and generate rewards (outputs), thus leading to a higher
    operational competitiveness. OCRA is thought of as an improvement of Data
    Envelopment Analysis (DEA): more efficient, robust, and properly sensitive
    to changes in inputs or outputs.

    In a general-purpose sense, PUs are the alternatives to be compared, and
    a quantity and value of each type of input/output is instead given as a
    criteria value and corresponding weight. Inputs are non-beneficial
    criteria that should be minimized, and Outputs are beneficial criteria
    that should be maximized; thus, the entire Decision Matrix is used.

    The performance of beneficial and non-beneficial criteria is computed
    separately and aggregated for each alternative, and then I/O criteria
    are summed to yield a final performance ranking. The Min value of each
    criteria is used in both cases (rather that Max - Min), following the
    implementation from Işık, A. T. (2016) and Madić, M. (2015). This means
    values are not scaled as 0-1 (possible future extension); however, they
    are floored to the Min value twice (first separately, then overall), such
    that the worst performance is always zero.

    .. math::

        I_i = \sum_{j=1}^{g} w_j \frac{\max{x_{ij}} - x_{ij}}{\min{x_{ij}}}\
        O_i = \sum_{j=g+1}^{n} w_j \frac{x_{ij} - \min{x_{ij}}}{\min{x_{ij}}}\
        for\ i = 1,2,3,...,m

    with:
    :math:`j = 1, 2, ..., g` for the objectives to be minimized,
    :math:`j = g + 1, g + 2, ..., n` for the objectives to be maximized.
    :math:`w_j` is the relative importance (weight) of each criteria.

    References
    ----------
    :cite:p:`parkan1994ocra`
    :cite:p:`tus2016ocra`
    :cite:p:`madic2015ocra`

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        rank, perf, perf_in, perf_out = ocra_performance(
            matrix,
            objectives,
            weights,
        )
        return rank, {
            "performance": perf,
            "input_performance": perf_in,
            "output_performance": perf_out,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "OCRA",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
