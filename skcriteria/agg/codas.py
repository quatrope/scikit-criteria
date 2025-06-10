#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Combinative Distance-Based Assessment - (CODAS)

The Multi-Criteria Decision Making (MCDM) problems have received a great variety
of methods and approaches developed within this field. This method aims
to use a Combinative Distance-Based Assessment to handle MCDM problems.

The concept of this method is based on computing the Euclidian distance and
the Taxicab distance to determine the desirability of an alternative.
The Euclidian distance is used as the primary measure and the Taxicab distance
as the secondary measure

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import itertools as it

    import numpy as np

    from scipy import stats

    from ._agg_base import KernelResult, RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, will_change


# =============================================================================
# EuclidianCODAS
# =============================================================================


class EuclidianCODAS(SKCDecisionMakerABC):
    #Descripcion piola de que hace

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError(
                "Error: CODAS can't operate with negative values on the DM Matrix"
            )
        if (weights > 1 or weights < 0 )
            raise ValueError(
                "Error: CODAS can only operate with weights between 0 and 1"
            ) # Step 3 del paper
        
        #ACA TENEMOS QUE LLAMAR AL METODO QUE IMPLEMENTAMOS
        #rank, score = fmf(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "EuclidianCODAS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


# =============================================================================
# TaxicabCODAS
# =============================================================================
class TaxicabCODAS(SKCDecisionMakerABC):

    #Descripcion piola de que hace
    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError(
                "Error: CODAS can't operate with negative values on the DM Matrix"
            )
        if (weights > 1 or weights < 0 )
            raise ValueError(
                "Error: CODAS can only operate with weights between 0 and 1"
            ) # Step 3 del paper
                
        #ACA TENEMOS QUE LLAMAR AL METODO QUE IMPLEMENTAMOS
        #rank, score = fmf(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "TaxicabCODAS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )