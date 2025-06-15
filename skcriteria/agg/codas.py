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

# TODO Limpiar imports

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
    # TODO Documentar
    n = len(euclidian_d)
    Ra = np.zeros((n, n))

    # TODO Revisar For Loop
    for i in range(n):
        for k in range(n):
            diff_E = euclidian_d[i] - euclidian_d[k]
            psi = 1 if abs(diff_E) >= tau else 0
            diff_T = taxicab_d[i] - taxicab_d[k]
            Ra[i, k] = diff_E + psi * diff_T

    return Ra

# TODO sacar objectives y weights y agregar tau
def codas(matrix, objectives, weights):
    # TODO Documentar

    # STEP4 Determinar la solucion negativa ideal
    ns_arr = np.min(matrix, axis=0)

    # STEP5 Calcular distancia manhattan y distancia euclidiana
    taxicab_distances = np.sum(np.abs(matrix - ns_arr), axis=1)

    euclidian_distances = np.sqrt(np.sum((matrix - ns_arr) ** 2, axis=1))

    # STEP6 Construir matriz de evaluacion relativa
    rel_assessment_m = codas_relative_assessment(
        euclidian_distances, taxicab_distances, tau=0.02
    )

    # STEP 7 Evaluar score de cada alternativa
    score = np.sum(rel_assessment_m, axis=1)

    return rank.rank_values(score, reverse=True), score


class CODAS(SKCDecisionMakerABC):
    # TODO Descripcion

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        # TODO Los valores tienen que estar entre 0 y 1
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
