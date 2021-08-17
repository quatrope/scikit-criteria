#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""SIMUS (Sequential Interactive Model for Urban Systems) Method"""


# =============================================================================
# IMPORTS
# =============================================================================

import operator
import warnings

import numpy as np

from ..base import SKCRankerMixin
from ..data import Objective, RankResult
from ..preprocessing import scale_by_sum
from ..utils import doc_inherit, rank, lp

# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

# STAGES ======================================================================


def _make_and_run_stage(transposed_matrix, b, senses, z_index, solver):
    # retrieve the problem class
    problem = (
        lp.Minimize if senses[z_index] == Objective.MIN.value else lp.Maximize
    )

    # create the variables
    xs = [
        lp.Float(f"x{idx}", low=0) for idx in range(transposed_matrix.shape[1])
    ]

    # create the objective function based on the criteria of row "z_index"
    stage_z_coefficients = transposed_matrix[z_index]
    stage_z = sum(
        coefficients * x for coefficients, x in zip(stage_z_coefficients, xs)
    )

    # create the stage
    stage = problem(z=stage_z, solver=solver)

    # the constraints are other files except the row of z_index
    for idx in range(transposed_matrix.shape[0]):
        if idx == z_index:
            continue
        coefficients = transposed_matrix[idx]

        # the two parts of the comparison
        left = sum(c * x for c, x in zip(coefficients, xs))
        right = b[idx]

        # >= if objective is to minimize <= maximize
        constraint = (
            (left >= right)
            if senses[idx] == Objective.MIN.value
            else (left <= right)
        )
        stage.subject_to(constraint)

    stage_result = stage.solve()
    return stage_result


def _solve_stages(transposed_matrix, b, objectives, solver):
    # execute the function inside the joblib environment one by objective.
    stages = []
    for idx in range(transposed_matrix.shape[0]):
        stage = _make_and_run_stage(
            transposed_matrix=transposed_matrix,
            b=b,
            senses=objectives,
            z_index=idx,
            solver=solver,
        )
        stages.append(stage)

    # create the results mtx
    arr_result = np.vstack([r.lp_values for r in stages])

    with np.errstate(invalid="ignore"):
        stages_result = scale_by_sum(arr_result, axis=1)

    # replace nan for 0
    stages_result[np.isnan(stages_result)] = 0

    return stages, stages_result


# FIRST METHOD  ===============================================================


def _first_method(*, stages_results):
    # project sum value
    sp = np.sum(stages_results, axis=0)

    # times that $v_{ij} > 0$ ($q$)
    q = np.sum(stages_results > 0, axis=0).astype(float)

    # participation factor
    fp = q / len(stages_results)

    # first method points
    vp = sp * fp

    return vp


# SECOND METHOD  ==============================================================


def _calculate_dominance_by_criteria(crit):
    shape = len(crit), 1
    crit_B = np.tile(crit, shape)
    crit_A = crit_B.T
    dominance = crit_A - crit_B
    dominance[dominance < 0] = 0
    return dominance


def _second_method(*, stages_results):
    # dominances by criteria
    dominance_by_criteria = []
    for crit in stages_results:
        dominance = _calculate_dominance_by_criteria(crit)
        dominance_by_criteria.append(dominance)

    # dominance
    dominance = np.sum(dominance_by_criteria, axis=0)

    # domination
    tita_j_p = np.sum(dominance, axis=1)

    # subordination
    tita_j_d = np.sum(dominance, axis=0)

    # second method score
    score = tita_j_p - tita_j_d

    return score, tita_j_p, tita_j_d, dominance, tuple(dominance_by_criteria)


# SIMUS =======================================================================


def simus(matrix, objectives, b=None, rank_by=1, solver="pulp"):

    transposed_matrix = matrix.T

    # check the b array and complete the missing values
    b = np.asarray(b)
    if None in b:
        mins = np.min(transposed_matrix, axis=1)
        maxs = np.max(transposed_matrix, axis=1)

        auto_b = np.where(objectives == Objective.MIN.value, mins, maxs)
        b = np.where(b != None, b, auto_b)

    # create and execute the stages
    stages, stages_results = _solve_stages(
        transposed_matrix=transposed_matrix,
        b=b,
        objectives=objectives,
        solver=solver,
    )

    # first method
    method_1_score = _first_method(stages_results=stages_results)

    # second method
    (
        method_2_score,
        tita_j_p,
        tita_j_d,
        dominance,
        dominance_by_criteria,
    ) = _second_method(stages_results=stages_results)

    # calculate ranking
    score = [method_1_score, method_2_score][rank_by - 1]
    ranking = rank(score, reverse=True)

    return (
        ranking,
        stages,
        stages_results,
        method_1_score,
        method_2_score,
        tita_j_p,
        tita_j_d,
        dominance,
        dominance_by_criteria,
    )


class SIMUS(SKCRankerMixin):
    def __init__(self, *, rank_by=1, solver="pulp"):
        self.solver = solver
        self.rank_by = rank_by

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        if not (
            isinstance(solver, lp.pulp.LpSolver)
            or lp.is_solver_available(solver)
        ):
            raise ValueError(f"solver {solver} not available")
        self._solver = solver

    @property
    def rank_by(self):
        return self._rank_by

    @rank_by.setter
    def rank_by(self, rank_by):
        if rank_by not in (1, 2):
            raise ValueError("'rank_by' must be 1 or 2")
        self._rank_by = rank_by

    @doc_inherit(SKCRankerMixin._validate_data)
    def _validate_data(self, objectives, weights, b, **kwargs):
        if len(np.unique(weights)) > 1:
            warnings.warn("SIMUS not take into account the weights")
        if b is not None and len(objectives) != len(b):
            raise ValueError("'b' must be the same leght as criteria or None")

    @doc_inherit(SKCRankerMixin._rank_data)
    def _rank_data(self, matrix, objectives, b, **kwargs):
        (
            ranking,
            stages,
            stages_results,
            method_1_score,
            method_2_score,
            tita_j_p,
            tita_j_d,
            dominance,
            dominance_by_criteria,
        ) = simus(
            matrix,
            objectives,
            b=b,
            rank_by=self.rank_by,
            solver=self.solver,
        )
        return ranking, {
            "rank_by": self._rank_by,
            "b": np.copy(b),
            "stages": stages,
            "stages_results": stages_results,
            "method_1_score": method_1_score,
            "method_2_score": method_2_score,
            "tita_j_p": tita_j_p,
            "tita_j_d": tita_j_d,
            "dominance": dominance,
            "dominance_by_criteria": dominance_by_criteria,
        }

    @doc_inherit(SKCRankerMixin._make_result)
    def _make_result(self, anames, rank, extra):
        return RankResult("SIMUS", anames=anames, rank=rank, extra=extra)

    def rank(self, dm, *, b=None):
        data = dm.to_dict()
        b = b if b is None else np.asarray(b)

        self._validate_data(b=b, **data)

        rank, extra = self._rank_data(b=b, **data)

        anames = data["anames"]
        result = self._make_result(anames=anames, rank=rank, extra=extra)

        return result
