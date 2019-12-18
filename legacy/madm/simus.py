#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# =============================================================================
# DOCS
# =============================================================================

"""SIMUS (Sequential Interactive Model for Urban Systems) Method"""

__all__ = [
    "SIMUS"]


# =============================================================================
# IMPORTS
# =============================================================================

import operator

import numpy as np

import joblib

from .. import norm, rank
from ..validate import MAX, MIN
from ..utils import lp
from ..utils.doc_inherit import doc_inherit

from ._dmaker import DecisionMaker


# =============================================================================
# FUNCTIONS
# =============================================================================

# ==============
# STAGES
# ==============

def _make_and_run_stage(mtx, b, senses, zindex, solver):
    # retrieve the problem class
    problem = lp.Minimize if senses[zindex] == MIN else lp.Maximize

    # create the variables
    xs = tuple(
        lp.Float("x{}".format(idx), low=0) for idx in range(mtx.shape[1]))

    # create the objective function
    z_coef = mtx[zindex]
    z = sum(c * x for c, x in zip(z_coef, xs))

    # the conditions
    conditions = []
    for idx in range(mtx.shape[0]):
        if idx == zindex:
            continue
        coef = mtx[idx]

        left = sum(c * x for c, x in zip(coef, xs))
        op = operator.le if senses[idx] == MAX else operator.ge
        right = b[idx]

        condition = op(left, right)
        conditions.append(condition)
    stage = problem(z=z, solver=solver).sa(*conditions)
    return stage, stage.solve()


def solve_stages(t_nmtx, b, ncriteria, solver, jobs):
    stages_results = jobs(
        joblib.delayed(_make_and_run_stage)(
            mtx=t_nmtx, b=b, senses=ncriteria, zindex=idx, solver=solver)
        for idx in range(t_nmtx.shape[0]))

    stages, results = [], []
    for s, r in stages_results:
        stages.append(s)
        results.append(r)

    # create the results mtx
    arr_result = np.vstack([np.asarray(r.values) for r in results])
    with np.errstate(invalid='ignore'):
        norm_result = norm.sum(arr_result, axis=1)
    norm_result[np.isnan(norm_result)] = 0

    return stages, norm_result


# ==============
# FIRST METHOD
# ==============

def first_method(stage_results):
    # project sum value
    sp = np.sum(stage_results, axis=0)

    # times that $v_{ij} > 0$ ($q$)
    q = np.sum(stage_results > 0, axis=0).astype(float)

    # participation factor
    fp = q / len(stage_results)

    # first method points
    vp = sp * fp

    return vp


# ==============
# SECOND METHOD
# ==============

def _dom_by_crit(crit):
    shape = len(crit), 1
    crit_B = np.tile(crit, shape)
    crit_A = crit_B.T
    dom = crit_A - crit_B
    dom[dom < 0] = 0
    return dom


def second_method(stage_results, jobs):
    # dominances by criteria
    dom_by_crit = jobs(
        joblib.delayed(_dom_by_crit)(crit)
        for crit in stage_results)

    # dominance
    doms = np.sum(dom_by_crit, axis=0)

    # domination
    tita_j_p = np.sum(doms, axis=1)

    # subordination
    tita_j_d = np.sum(doms, axis=0)

    # second method points
    points = tita_j_p - tita_j_d

    return points, tita_j_p, tita_j_d, doms, tuple(dom_by_crit)


# ===============
# SIMUS FUNCTION
# ===============

def simus(
    nmtx, ncriteria, nweights,
    rank_by=1, b=None, solver="pulp", njobs=None
):
    # determine the njobs
    njobs = njobs or joblib.cpu_count()

    t_nmtx = nmtx.T

    # check the b array and complete the missing values
    b = np.asarray(b)
    if None in b:
        mins = np.min(t_nmtx, axis=1)
        maxs = np.max(t_nmtx, axis=1)

        auto_b = np.where(ncriteria == MAX, maxs, mins)
        b = np.where(b.astype(bool), b, auto_b)

    # multiprocessing environment
    with joblib.Parallel(n_jobs=njobs) as jobs:

        # create and execute the stages
        stages, stage_results = solve_stages(
            t_nmtx=t_nmtx, b=b, ncriteria=ncriteria,
            solver=solver, jobs=jobs)

        # first methods points
        points1 = first_method(stage_results)
        points2, tita_j_p, tita_j_d, doms, dom_by_crit = second_method(
            stage_results, jobs)

    points = [points1, points2][rank_by - 1]
    ranking = rank.rankdata(points, reverse=True)

    return (
        ranking, stages, stage_results, points1,
        points2, tita_j_p, tita_j_d, doms, dom_by_crit)


# =============================================================================
# OO
# =============================================================================

class SIMUS(DecisionMaker):
    r"""SIMUS (Sequential Interactive Model for Urban Systems) developed
    by Nolberto Munier (2011) is a tool to aid decision-making problems with
    multiple objectives. The method solves successive scenarios formulated as
    linear programs. For each scenario, the decision-maker must choose the
    criterion to be considered objective while the remaining restrictions
    constitute the constrains system that the projects are subject to. In each
    case, if there is a feasible solution that is optimum, it is recorded in a
    matrix of efficient results. Then, from this matrix two rankings allow the
    decision maker to compare results obtained by different procedures.
    The first ranking is obtained through a linear weighting of each column by
    a factor - equivalent of establishing a weight - and that measures the
    participation of the corresponding project. In the second ranking, the
    method uses dominance and subordinate relationships between projects,
    concepts from the French school of MCDM.

    Parameters
    ----------

    mnorm : string, callable, optional (default="none")
        Normalization method for the alternative matrix.

    wnorm : string, callable, optional (default="none")
        Normalization method for the weights array.

    rank_by : 1 or 2 (default=1)
        Wich of the two methods are used to calculate the ranking.
        The two methods are executed always.

    solver : str, default="pulp"
        Which solver to use to solve the undelying linear programs. The full
        list are available in `skcriteria.utils.lp.SOLVERS`

    njobs : int, default=None
        How many cores to use to solve the linear programs and the second
        method. By default all the availables cores are used.

    Returns
    -------

     Decision : :py:class:`skcriteria.madm.Decision`
        With values:

        - **kernel_**: None
        - **rank_**: A ranking (start at 1) where the i-nth element represent
          the position of the i-nth alternative.
        - **best_alternative_**: The index of the best alternative.
        - **alpha_solution_**: True
        - **beta_solution_**: False
        - **gamma_solution_**: True
        - **e_**: Particular data created by this method.

          - **rank_by**: 1 or 2. Wich of the two methods are used to
            calculate the ranking. Esentialy if the rank is calculated with
            ``e_.points1`` or ``e_points2``
          - **solver**: With solver was used for the underlying linear
            problems.
          - **stages**: The underlying linear problems.
          - **stage_results**: The values of the variables of the linear
            problems as a n-dimensional array. When th `n-th` row represent
            the result values of the variables for the `n-th` stage.
          - **points1**: The points of every alternative obtained by the
            first method.
          - **points2**: The points of every alternative obtained by the
            first method.
          - **tita_j_p**: 2nd. method domination.
          - **tita_j_d**: 2nd. method subordination.
          - **doms**: Total dominance matrix of the 2nd. method.
          - **dom_by_crit**: Dominance by criteria of the 2nd method.

    References
    ----------

    .. [1] Munier, N. (2011). A strategy for using multicriteria analysis in
       decision-making: a guide for simple and complex environmental projects.
       Springer Science & Business Media.
    .. [2] Munier, N., Carignano, C., & Alberto, C. UN MÉTODO DE PROGRAMACIÓN
       MULTIOBJETIVO. Revista de la Escuela de Perfeccionamiento en
       Investigación Operativa, 24(39).


    """

    def __init__(self, mnorm="none", wnorm="none",
                 rank_by=1, solver="pulp", njobs=None):
        super(SIMUS, self).__init__(mnorm=mnorm, wnorm=wnorm)
        self._solver = solver
        self._njobs = njobs
        self._rank_by = rank_by

    @doc_inherit
    def solve(self, ndata, b):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        data = simus(
            nmtx, ncriteria, nweights, rank_by=self._rank_by,
            b=b, solver=self._solver, njobs=self._njobs)
        (ranking, stages, stage_results, points1,
         points2, tita_j_p, tita_j_d, doms, dom_by_crit) = data
        ranking = data[0]
        extra = {
            "rank_by": self._rank_by,
            "solver": self._solver,
            "stages": data[1],
            "stage_results": data[2],
            "points1": data[3],
            "points2": data[4],
            "tita_j_p": data[5],
            "tita_j_d": data[6],
            "doms": data[7],
            "dom_by_crit": data[8]}
        return None, ranking, extra

    @property
    def solver(self):
        """Which solver to use to solve the undelying linear programs. The full
        list are available in `skcriteria.utils.lp.SOLVERS`

        """
        return self._solver

    @property
    def njobs(self):
        """How many cores to use to solve the linear programs and the second
        method. By default all the availables cores are used.

        """
        return self._njobs


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
