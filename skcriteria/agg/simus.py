#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""SIMUS (Sequential Interactive Model for Urban Systems) Method."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import warnings

    import numpy as np

    from scipy.optimize import linprog

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..preprocessing.scalers import scale_by_sum
    from ..utils import Bunch, doc_inherit, rank


# =============================================================================
# CONSTANTS
# =============================================================================

#: Available solvers (``method`` of :func:`scipy.optimize.linprog`).
_SOLVERS = ("highs", "highs-ds", "highs-ipm")

#: Names of the ``scipy.optimize.linprog`` status codes.
_LINPROG_STATUS = {
    0: "Optimal",
    1: "Not Solved",  # iteration/time limit
    2: "Infeasible",
    3: "Unbounded",
    4: "Undefined",  # numerical difficulties
}

#: Relative tolerances used to decide if a stage has multiple optima.
_OBJ_RTOL = 1e-9
_X_RTOL = 1e-6


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

# STAGES ======================================================================


def _stage_matrices(transposed_matrix, b, senses, z_index):
    """Build the matrix form ``min c @ x s.t. A_ub @ x <= b_ub, x >= 0``.

    Row ``z_index`` is the objective; every other row is a constraint with
    ``>=`` if its criterion is minimized and ``<=`` if it is maximized. The
    ``>=`` rows and the objective of a MAX stage are negated so the problem
    is always expressed as a minimization with ``<=`` constraints.

    """
    n_crit = transposed_matrix.shape[0]
    is_max = senses[z_index] == Objective.MAX.value

    c = transposed_matrix[z_index] * (-1.0 if is_max else 1.0)

    rows = np.array([idx for idx in range(n_crit) if idx != z_index], int)
    sign = np.where(senses[rows] == Objective.MIN.value, -1.0, 1.0)
    A_ub = transposed_matrix[rows] * sign[:, np.newaxis]
    b_ub = b[rows] * sign

    return c, A_ub, b_ub, is_max


def _equilibrate(c, A_ub, b_ub):
    """Divide the objective and every constraint by its max absolute value.

    Positive row scaling changes neither the feasible set nor the optimal
    solutions, but it keeps the problem inside the absolute tolerances of
    the solver when the criteria have very different magnitudes (e.g.
    ``1e-8`` vs ``1e11``); without it the solver may stop at
    suboptimal vertices.

    """
    c_scale = np.max(np.abs(c), initial=0.0) or 1.0
    row_scale = np.max(np.abs(A_ub), axis=1, initial=0.0)
    row_scale[row_scale == 0] = 1.0
    return (
        c / c_scale,
        A_ub / row_scale[:, np.newaxis],
        b_ub / row_scale,
        c_scale,
    )


def _optimal_face_ranges(c, A_ub, b_ub, res):
    """Detect alternative optima of a solved stage.

    Returns ``(multiple_optima, ranges)`` where ``ranges`` is ``None`` when
    the optimum is proven unique by the cheap rank test, or an ``(n, 2)``
    array with the ``[min, max]`` that every variable can take over the
    optimal face otherwise.

    The optimal face is ``{x >= 0, s >= 0 : A_ub x + s = b_ub}`` restricted
    to the variables whose reduced cost (``x``) or dual (``s``) is zero
    (complementary slackness with the dual returned by HiGHS). If the columns
    of those free variables are linearly independent the optimum is unique.
    Otherwise the face is explored with two auxiliary LPs (min/max of
    ``x_j`` with the objective fixed at its optimal value) for every ``x_j``
    with zero reduced cost; the others are ``0`` over the whole face.

    """
    n_vars, n_cons = len(c), len(b_ub)
    scale = max(1.0, np.max(np.abs(c), initial=0.0))
    dual_tol = 1e-9 * scale

    free_x = np.abs(res.lower.marginals) <= dual_tol
    if n_cons:
        free_s = np.abs(res.ineqlin.marginals) <= dual_tol
        M = np.hstack([A_ub, np.eye(n_cons)])
        free = np.concatenate([free_x, free_s])
    else:
        M = np.zeros((0, n_vars))
        free = free_x

    M_free = M[:, free]
    if M_free.shape[1] == 0 or (
        M_free.shape[0] >= M_free.shape[1]
        and np.linalg.matrix_rank(M_free) == M_free.shape[1]
    ):
        return False, None

    # the rank test is inconclusive: range the free variables over the face
    obj_tol = _OBJ_RTOL * max(1.0, abs(res.fun))
    A_face = np.vstack([A_ub, c]) if n_cons else c[np.newaxis, :]
    b_face = np.append(b_ub, res.fun + obj_tol)
    bounds = [(0, None) if free_j else (0, 0) for free_j in free_x]

    ranges = np.zeros((n_vars, 2))
    for j in np.flatnonzero(free_x):
        e_j = np.zeros(n_vars)
        e_j[j] = 1.0
        for col, direction in ((0, 1.0), (1, -1.0)):
            aux = linprog(
                direction * e_j,
                A_ub=A_face,
                b_ub=b_face,
                bounds=bounds,
                method="highs",
            )
            # the face is never empty (res.x is in it) so the only possible
            # failure is an unbounded max
            ranges[j, col] = direction * aux.fun if aux.status == 0 else np.inf

    x_tol = _X_RTOL * max(1.0, np.max(np.abs(res.x)))
    multiple_optima = bool(np.any(ranges[:, 1] - ranges[:, 0] > x_tol))
    return multiple_optima, ranges


def _make_and_run_stage(transposed_matrix, b, senses, z_index, solver):
    """Solve one SIMUS stage with :func:`scipy.optimize.linprog`.

    ``lp_status_code`` is the ``linprog`` status code and ``lp_status`` its
    name. ``lp_multiple_optima`` and ``lp_values_range`` report if the
    optimal solution is unique (see :func:`_optimal_face_ranges`).

    """
    c, A_ub, b_ub, is_max = _stage_matrices(
        transposed_matrix, b, senses, z_index
    )
    n_vars = len(c)

    # the scaled problem is the one solved and checked for degeneracy; the
    # original one is reported in "lp_problem"
    sc_c, sc_A_ub, sc_b_ub, c_scale = _equilibrate(c, A_ub, b_ub)

    res = linprog(
        sc_c,
        A_ub=sc_A_ub if len(sc_b_ub) else None,
        b_ub=sc_b_ub if len(sc_b_ub) else None,
        bounds=(0, None),
        method=solver,
    )
    status_code, status = res.status, _LINPROG_STATUS[res.status]

    if res.status == 0:
        values = np.asarray(res.x, dtype=float)
        objective = float(res.fun * c_scale)
        objective = -objective if is_max else objective
        multiple_optima, values_range = _optimal_face_ranges(
            sc_c, sc_A_ub, sc_b_ub, res
        )
    else:
        # no optimal solution (e.g. unbounded): the stage does not
        # contribute to the efficient matrix
        values = np.zeros(n_vars)
        objective = None
        multiple_optima, values_range = False, None

    problem = Bunch(
        "lp_problem",
        {
            "sense": "max" if is_max else "min",
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "bounds": (0, None),
        },
    )
    stage_result = Bunch(
        "result",
        {
            "lp_status_code": status_code,
            "lp_status": status,
            "lp_objective": objective,
            "lp_variables": np.array([f"x{i}" for i in range(n_vars)]),
            "lp_values": values,
            "lp_multiple_optima": multiple_optima,
            "lp_values_range": values_range,
            "lp_problem": problem,
            "lp_solver_result": res,
        },
    )

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


def simus(matrix, objectives, b=None, rank_by=1, solver="highs"):
    """Execute SIMUS without any validation."""
    transposed_matrix = matrix.T

    # check the b array and complete the missing values
    b = np.asarray(b)
    if None in b:
        mins = np.min(transposed_matrix, axis=1)
        maxs = np.max(transposed_matrix, axis=1)

        auto_b = np.where(objectives == Objective.MIN.value, mins, maxs)
        b = np.where(b != None, b, auto_b)  # noqa
    b = np.asarray(b, dtype=float)

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
    solver : str, (default="highs")
        Which solver to use to solve the underlying linear programs, passed
        as ``method`` to :func:`scipy.optimize.linprog`. One of ``"highs"``
        (HiGHS chooses the algorithm), ``"highs-ds"`` (dual simplex) or
        ``"highs-ipm"`` (interior point with crossover). None is an alias of
        ``"highs"``. All of them return the same optimal objective values;
        when a stage has multiple optima they may return different optimal
        solutions (and therefore different rankings), which is reported with
        a warning. The legacy ``linprog`` methods are not supported because
        they do not return the dual values needed to detect multiple optima.

    Warnings
    --------
    UserWarning:
        If the method detect different weights by criteria.
    UserWarning:
        If some stage LP has multiple optimal solutions (the efficient
        matrix, and therefore the ranking, depends on the vertex returned by
        the solver) or has no optimal solution (e.g. it is unbounded).

    Raises
    ------
    ValueError:
        If the length of b does not match the number of criteria, if
        ``rank_by`` is not 1 or 2, or if the solver is not available.

    See
    ---
    :func:`scipy.optimize.linprog`

    """

    _skcriteria_parameters = ["rank_by", "solver"]

    def __init__(self, *, rank_by=1, solver="highs"):
        solver = "highs" if solver is None else solver
        if solver not in _SOLVERS:
            raise ValueError(
                f"solver {solver!r} not available. Use one of {_SOLVERS}"
            )
        self._solver = solver
        if rank_by not in (1, 2):
            raise ValueError("'rank_by' must be 1 or 2")
        self._rank_by = rank_by

    @property
    def solver(self):
        """Solver used to solve the stage LPs."""
        return self._solver

    @property
    def rank_by(self):
        """Which of the two ranking provided by SIMUS is used."""
        return self._rank_by

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(
        self, matrix, objectives, b, weights, criteria=None, **kwargs
    ):
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

        criteria = np.arange(len(objectives)) if criteria is None else criteria
        multiple_optima = np.array(
            [bool(s.lp_multiple_optima) for s in stages]
        )
        if multiple_optima.any():
            warnings.warn(
                "SIMUS: the LP stages with criteria "
                f"{list(np.asarray(criteria)[multiple_optima])} as objective "
                "have multiple optimal solutions. The efficient matrix (and "
                "potentially the ranking) depends on the vertex chosen by the "
                "solver. See 'stages[i].lp_values_range' in the result extra.",
                UserWarning,
                stacklevel=4,
            )
        not_optimal = [
            (crit, s.lp_status)
            for crit, s in zip(criteria, stages)
            if s.lp_status != "Optimal"
        ]
        if not_optimal:
            warnings.warn(
                "SIMUS: some LP stages have no optimal solution "
                f"(criterion, status): {not_optimal}. Their row in the "
                "efficient matrix is set to 0.",
                UserWarning,
                stacklevel=4,
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
            "multiple_optima": multiple_optima,
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
        b = b if b is None else np.asarray(b)
        return self._evaluate_dm(dm, b=b)
