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

import warnings

with hidden():

    import numpy as np


    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank


# =============================================================================
# CODAS
# =============================================================================



def codas_relative_assessment(euclidian_d, taxicab_d, tau):
    """Aux function to construct the relative assessment matrix, used for final ranking  """
    E_i = euclidian_d[:, np.newaxis]  
    E_k = euclidian_d[np.newaxis, :]  

    T_i = taxicab_d[:, np.newaxis]
    T_k = taxicab_d[np.newaxis, :]

    # Diferencias
    diff_E = E_i - E_k
    diff_T = T_i - T_k

    # Condición psi: |diff_E| >= tau --> 1, si no --> 0
    psi = (np.abs(diff_E) >= tau).astype(int)

    #Matriz final
    Rel_ass_matrix = diff_E + psi * diff_T

    return Rel_ass_matrix

def codas(matrix, tau):
    """Execute CODAS without any validation and assuming tau value."""

    # STEP4 Determinar la solucion negativa ideal
    ns_arr = np.min(matrix, axis=0)

    # STEP5 Calcular distancia manhattan y distancia euclidiana
    taxicab_distances = np.sum(np.abs(matrix - ns_arr), axis=1)

    euclidian_distances = np.sqrt(np.sum((matrix - ns_arr) ** 2, axis=1))

    # STEP6 Construir matriz de evaluacion relativa
    rel_assessment_m = codas_relative_assessment(
        euclidian_distances, taxicab_distances, tau
    )

    # STEP 7 Evaluar score de cada alternativa
    score = np.sum(rel_assessment_m, axis=1)

    return rank.rank_values(score, reverse=True), score


class CODAS(SKCDecisionMakerABC):
    """Rank alternatives using CODAS method.

    COmbinative Distance-based ASsessment (CODAS) is a
    method to handle MCDM problems and rank the alternatives

    The concept of this method is based on computing the Euclidean distance
    and the Taxicab distance in order to determine the desirability of an alternative
    The Euclidean distance is used as a primary measure, 
    while the Taxicab distance as a secondary one.


    Parameters
    ----------
    tau : float, optional (default=0.02)
        τ is the threshold parameter that can be set by the decision-maker.
        used to construct the relative assessment matrix.
        ψ is a threshold function that uses tau to recognize the equality
        of the Euclidean distances

    Warnings
    --------
    UserWarning:
        If tau is not set between 0.02 and 0.05.
    """

    _skcriteria_parameters = ["tau"]

    def __init__(self, tau=0.02):
        self._tau = tau

    @property
    def tau(self):
        """Which tau value will be used."""
        return self._tau

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        if np.any(matrix > 1):
            raise ValueError(
                "Error: DM Matrix must be normalized (Use codas Transformer)"
            )
        if np.any(matrix < 0):
            raise ValueError(
                "Error: CODAS can't operate with negative values on the DM Matrix"
            )
        if self.tau < 0.01 or self.tau > 0.05:
            warnings.warn("It is suggested to set tau at a value between 0.01 and 0.05")
        rank, score = codas(matrix, tau=self.tau)
        return rank, {"score": score}

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "CODAS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
