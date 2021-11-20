#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""SIMUS (Sequential Interactive Model for Urban Systems) Method."""


# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

from ..core import Objective, RankResult, SKCDecisionMakerABC
from ..preprocessing.scalers import scale_by_sum
from ..utils import doc_inherit, lp, rank

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
    """Execute SIMUS without any validation."""
    transposed_matrix = matrix.T

    # check the b array and complete the missing values
    b = np.asarray(b)
    if None in b:
        mins = np.min(transposed_matrix, axis=1)
        maxs = np.max(transposed_matrix, axis=1)

        auto_b = np.where(objectives == Objective.MIN.value, mins, maxs)
        b = np.where(b != None, b, auto_b)  # noqa

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
    ranking = rank.rank_values(score, reverse=True)

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


class SIMUS(SKCDecisionMakerABC):
    r"""SIMUS (Sequential Interactive Model for Urban Systems).

    SIMUS developed by Nolberto Munier (2011) is a tool to aid decision-making
    problems with multiple objectives. The method solves successive scenarios
    formulated as linear programs. For each scenario, the decision-maker must
    choose the criterion to be considered objective while the remaining
    restrictions constitute the constrains system that the projects are subject
    to. In each case, if there is a feasible solution that is optimum, it is
    recorded in a matrix of efficient results. Then, from this matrix two
    rankings allow the decision maker to compare results obtained by different
    procedures. The first ranking is obtained through a linear weighting of
    each column by a factor - equivalent of establishing a weight - and that
    measures the participation of the corresponding project. In the second
    ranking, the method uses dominance and subordinate relationships between
    projects, concepts from the French school of MCDM.

    Parameters
    ----------
    rank_by : 1 or 2 (default=1)
        Witch of the two methods are used to calculate the ranking.
        The two methods are executed always.
    solver : str, (default="pulp")
        Which solver to use to solve the underlying linear programs. The full
        list are available in `pulp.listSolvers(True)`. "pulp" or None used
        the default solver selected by "PuLP".

    Warnings
    --------
    UserWarning:
        If the method detect different weights by criteria.

    Raises
    ------
    ValueError:
        If the length of b does not match the number of criteria.

    See
    ---
    `PuLP Documentation <https://coin-or.github.io/pulp/>`_

    """

    def __init__(self, *, rank_by=1, solver="pulp"):
        self.solver = solver
        self.rank_by = rank_by

    @property
    def solver(self):
        """Solver used by PuLP."""
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
        """Which of the two ranking provided by SIMUS is used."""
        return self._rank_by

    @rank_by.setter
    def rank_by(self, rank_by):
        if rank_by not in (1, 2):
            raise ValueError("'rank_by' must be 1 or 2")
        self._rank_by = rank_by

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, b, weights, **kwargs):
        if len(np.unique(weights)) > 1:
            warnings.warn("SIMUS not take into account the weights")
        if b is not None and len(objectives) != len(b):
            raise ValueError("'b' must be the same leght as criteria or None")
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

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "SIMUS", alternatives=alternatives, values=values, extra=extra
        )

    def evaluate(self, dm, *, b=None):
        """Validate the decision matrix and calculate a ranking.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.
        b: :py:class:`numpy.ndarray`
            Right-side-value of the LP problem,

            SIMUS automatically assigns the vector of the right side (b) in
            the constraints of linear programs.

            If the criteria are to maximize, then the constraint is <=;
            and if the column minimizes the constraint is >=.
            The b/right side value limits of the constraint are chosen
            automatically based on the minimum or maximum value of the
            criteria/column if the constraint is <= or >= respectively.

            The user provides "b" in some criteria and lets SIMUS choose
            automatically others.  For example, if you want to limit the two
            constraints of the dm with 4 criteria by the value 100,  b must be
            `[None, 100, 100, None]` where None will be chosen automatically
            by SIMUS.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.

        """
        data = dm.to_dict()
        b = b if b is None else np.asarray(b)

        rank, extra = self._evaluate_data(b=b, **data)

        alternatives = data["alternatives"]
        result = self._make_result(
            alternatives=alternatives, values=rank, extra=extra
        )

        return result
