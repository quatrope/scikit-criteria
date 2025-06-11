#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

""" The proposed method that is called EDAS (Evaluation based
on Distance from Average Solution) uses average solution for appraising the alternatives
(inventory items). Two measures which called PDA (positive distance from average) and
NDA (negative distance from average) are considered for the appraisal in this study. These
measures are calculated according to the type of criteria (beneficial or non-beneficial).

This method is very useful when we have some conflicting criteria.
The desirable alternative has lower distance from ideal solution and higher distance 
from nadir solution in these MCDM methods. However, the best alternative in the 
proposed method is related to the distance from average solution (AV )."""

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
# EDAS
# =============================================================================

def edas(matrix, objectives):
    decision_matrix = np.array(matrix)

    rows = np.shape(decision_matrix)[0]
    columns = np.shape(decision_matrix)[1]

    average_solution = np.mean(decision_matrix, axis=0)




class EDAS(SKDecisionMakesABC):
    r""" Descripcion facha """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives):
        if np.any(matrix < 0):
            raise ValueError("Edas can't operate with value < 0")
        
        rank, score = edas(matrix, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "WeightedSumModel",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )