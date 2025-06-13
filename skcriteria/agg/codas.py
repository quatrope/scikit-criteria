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
    from ..utils import doc_inherit, rank


# =============================================================================
# CODAS
# =============================================================================

def codas_relative_assessment(euclidian_d, taxicab_d, tau=0.02):
    n= len(euclidian_d)
    Ra= np.zeros((n,n))

    for i in range(n):
        for k in range(n):
            diff_E = euclidian_d[i] - euclidian_d[k]
            psi = 1 if abs(diff_E)>= tau else 0
            diff_T= taxicab_d[i]- taxicab_d[k]
            Ra[i,k] = diff_E + psi * diff_T
    
    return Ra




def codas(matrix, objectives, weights):
    ##STEP2 : matriz normalizada
    norm_matrix = np.zeros_like(matrix, dtype=float)
    if  Objective.MAX.value in objectives:
        max_columns= matrix[: , objectives == Objective.MAX.value]
        #BUG:might be zero?
        max_values = np.max(max_columns, axis=0)
        #Ecuacion (1)
        norm_matrix[:, objectives == Objective.MAX.value] = (max_columns / max_values)

    if Objective.MIN.value in objectives:
        min_columns = matrix[:,objectives == Objective.MIN.value]
        #BUG:might be zero?
        min_values = np.min(min_columns, axis=0)
        #Ecuacion (1)
        norm_matrix[:, objectives == Objective.MIN.value] = (min_values / min_columns)

    

    ##STEP3 matriz normalizada con pesos
    w_norm_matrix = np.multiply(norm_matrix, weights)


    ##STEP4 Determinar la solucion negativa ideal
    ns_arr = np.min(w_norm_matrix, axis=0)

    #STEP5 Calcular distancia manhattan y distancia euclidiana
    taxicab_distances = np.sum(np.abs(w_norm_matrix - ns_arr), axis=1)

    euclidian_distances = np.sqrt(np.sum((w_norm_matrix - ns_arr)**2, axis=1 ))

    #STEP6 construir matriz de evaluacion relativa
    rel_assessment_m = codas_relative_assessment(euclidian_distances, taxicab_distances, tau=0.02)

    #Evaluar score de cada alternativa
    score = np.sum(rel_assessment_m, axis=1)
    return rank.rank_values(score, reverse=True), score

class CODAS(SKCDecisionMakerABC):
    #Descripcion piola de que hace

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix <= 0):
            raise ValueError(
                "Error: CODAS can't operate with negative values on the DM Matrix"
            )
        rank, score = codas(matrix, objectives, weights)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "CODAS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


