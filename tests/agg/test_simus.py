#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.simus."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
import skcriteria.datasets
from skcriteria.agg import RankResult
from skcriteria.agg.simus import SIMUS

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_SIMUS_munier24metodo():
    """
    Data From:
        Munier, N., Carignano, C., & Alberto, C.
        UN MÉTODO DE PROGRAMACIÓN MULTIOBJETIVO.
        Revista de la Escuela de Perfeccionamiento en Investigación
        Operativa, 24(39).
    """

    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
        alternatives=["Proyecto 1", "Proyecto 2", "Proyecto 3"],
        criteria=["Criterio 1", "Criterio 2", "Criterio 3", "Criterio 4"],
    )
    b = [None, 500, None, None]

    expected = RankResult(
        "SIMUS",
        ["Proyecto 1", "Proyecto 2", "Proyecto 3"],
        [3, 2, 1],
        {
            "method_1_score": [0.09090909, 0.66603535, 0.74305556],
            "method_2_score": [-2.45454545, 0.99621211, 1.45833334],
            "rank_by": 1,
        },
    )

    ranker = SIMUS()
    result = ranker.evaluate(dm, b=b)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.b == b)
    assert np.all(result.e_.rank_by == expected.e_.rank_by)
    assert np.allclose(result.e_.method_1_score, expected.e_.method_1_score)
    assert np.allclose(result.e_.method_2_score, expected.e_.method_2_score)


def test_SIMUS_solver_not_available():
    with pytest.raises(ValueError):
        SIMUS(solver="fooo")


def test_SIMUS_solver_incorrect_rank_by():
    with pytest.raises(ValueError):
        SIMUS(rank_by=3)


def test_SIMUS_multiple_weights_warning():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
        weights=[1, 2, 3, 4],
    )

    ranker = SIMUS()
    with pytest.warns(UserWarning):
        ranker.evaluate(dm)


def test_SIMUS_incorrect_b():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
    )
    b = [1, 2, 3]

    ranker = SIMUS()
    with pytest.raises(ValueError):
        ranker.evaluate(dm, b=b)


# =============================================================================
# HIGHS BACKEND
# =============================================================================


def _munier24metodo():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
    )
    return dm, [None, 500, None, None]


def test_SIMUS_default_solver_is_highs():
    assert SIMUS().solver == "highs"
    assert SIMUS(solver=None).solver == "highs"


@pytest.mark.parametrize(
    "solver", ["simplex", "revised simplex", "interior-point"]
)
def test_SIMUS_legacy_linprog_methods_not_available(solver):
    with pytest.raises(ValueError):
        SIMUS(solver=solver)


@pytest.mark.parametrize("solver", ["highs", "highs-ds", "highs-ipm"])
def test_SIMUS_highs_solvers_munier24metodo(solver):
    dm, b = _munier24metodo()
    reference = SIMUS().evaluate(dm, b=b)
    result = SIMUS(solver=solver).evaluate(dm, b=b)

    assert result.e_.stages[0].lp_solver_result is not None
    assert result.values_equals(reference)
    assert np.allclose(
        result.e_.stages_results, reference.e_.stages_results, atol=1e-7
    )
    assert np.allclose(
        [s.lp_objective for s in result.e_.stages],
        [s.lp_objective for s in reference.e_.stages],
    )


@pytest.mark.parametrize("solver", ["highs", "highs-ds", "highs-ipm"])
def test_SIMUS_highs_solvers_same_objectives_on_multiple_optima(solver):
    # alternatives 0 and 1 are clones: the optimal objective is unique but
    # the optimal solution is not, whatever the algorithm
    dm = skcriteria.mkdm(
        matrix=[[5, 3, 2], [5, 3, 2], [2, 6, 4], [1, 2, 9]],
        objectives=[max, max, min],
    )
    with pytest.warns(UserWarning, match="multiple"):
        result = SIMUS(solver=solver).evaluate(dm)

    assert result.e_.multiple_optima.tolist() == [True, False, False]
    assert np.isclose(result.e_.stages[0].lp_objective, 10.0)
    assert np.isclose(result.e_.stages[0].lp_values[:2].sum(), 2.0)


def test_SIMUS_munier24metodo_efficient_matrix():
    dm, b = _munier24metodo()
    result = SIMUS().evaluate(dm, b=b)

    expected_objectives = [540.0, 377.94117647, 0.0, 2529.06976744]
    expected_stages_results = [
        [0.125, 0.0, 0.875],
        [0.0, 0.38888889, 0.61111111],
        [0.0, 0.0, 0.0],
        [0.05681818, 0.94318182, 0.0],
    ]

    objectives = [s.lp_objective for s in result.e_.stages]
    assert np.allclose(objectives, expected_objectives)
    assert np.allclose(result.e_.stages_results, expected_stages_results)
    assert not result.e_.multiple_optima.any()
    assert all(s.lp_status == "Optimal" for s in result.e_.stages)


@pytest.mark.parametrize("rank_by", [1, 2])
def test_SIMUS_simple_stock_selection(rank_by):
    dm = skcriteria.datasets.load_simple_stock_selection()
    result = SIMUS(rank_by=rank_by).evaluate(dm)

    assert result.values_equals(
        RankResult("SIMUS", dm.alternatives, [1, 2, 2, 2, 1, 2], {})
    )
    assert np.allclose(
        [s.lp_objective for s in result.e_.stages], [11.2, 49.0, 0.0]
    )
    assert np.allclose(
        result.e_.stages_results,
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    assert np.allclose(result.e_.method_1_score, [1 / 3, 0, 0, 0, 1 / 3, 0])
    assert np.allclose(result.e_.method_2_score, [4, -2, -2, -2, 4, -2])
    assert not result.e_.multiple_optima.any()


@pytest.mark.parametrize(
    "windows_size, objectives, stages_results",
    [
        (
            7,
            [
                6.902752293578e-02,
                3.821995677233e-03,
                2.392828399123e11,
                3.014164772727e09,
                1.407461538462e00,
                1.514966840238e-08,
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.69181077, 0.0, 0.30818923, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.23027993, 0.76972007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
        (
            15,
            [
                1.651815286624e-01,
                6.065514858328e-03,
                5.285921786265e11,
                6.383127272727e09,
                1.362193548387e00,
                2.539377268378e-08,
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.66228227, 0.0, 0.33771773, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.23660923, 0.76339077, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_SIMUS_van2021evaluation(windows_size, objectives, stages_results):
    """The 'xm' criterion is ~1e-8 while others are ~1e11.

    Without row scaling the stage with 'xm' as objective falls inside the
    absolute tolerances of the solver, which stops at a suboptimal vertex
    (objective ~1e-13 or ~1e-9 instead of ~1e-8).

    """
    dm = skcriteria.datasets.load_van2021evaluation(windows_size)
    result = SIMUS().evaluate(dm)

    assert result.values_equals(
        RankResult("SIMUS", dm.alternatives, [6, 5, 1, 3, 4, 6, 6, 2, 6], {})
    )
    assert not result.e_.multiple_optima.any()
    assert all(s.lp_status == "Optimal" for s in result.e_.stages)
    assert np.allclose(
        [s.lp_objective for s in result.e_.stages], objectives, rtol=1e-7
    )
    assert np.allclose(result.e_.stages_results, stages_results, atol=1e-7)

    # the reported objective matches the original (unscaled) LP
    for stage in result.e_.stages:
        value = stage.lp_problem.c @ stage.lp_values
        value = -value if stage.lp_problem.sense == "max" else value
        assert np.isclose(value, stage.lp_objective, rtol=1e-7)


def test_SIMUS_values_follow_alternatives_order():
    # with n >= 11 alternatives the values must follow the order of the
    # alternatives (x0, x1, ..., x11), not the lexicographic order of the
    # variable names (x0, x1, x10, x11, x2, ...)
    rng = np.random.default_rng(0)
    matrix = rng.uniform(1, 100, (12, 3))
    dm = skcriteria.mkdm(matrix, [max, max, min])

    result = SIMUS().evaluate(dm)

    assert not result.e_.multiple_optima.any()
    for stage in result.e_.stages:
        assert list(stage.lp_variables) == [f"x{i}" for i in range(12)]
    assert np.argmax(result.e_.stages_results[0]) == 4
    assert np.argmax(result.e_.stages_results[1]) == 1
    x = result.e_.stages[0].lp_values
    assert np.isclose(matrix[:, 0] @ x, result.e_.stages[0].lp_objective)


def test_SIMUS_multiple_optima_detected():
    # alternatives 0 and 1 are clones: in the first stage any split of the
    # production between them is optimal
    dm = skcriteria.mkdm(
        matrix=[[5, 3, 2], [5, 3, 2], [2, 6, 4], [1, 2, 9]],
        objectives=[max, max, min],
        criteria=["C0", "C1", "C2"],
    )
    with pytest.warns(UserWarning, match=r"criteria \['C0'\].*multiple"):
        result = SIMUS().evaluate(dm)

    assert result.e_.multiple_optima.tolist() == [True, False, False]

    stage = result.e_.stages[0]
    assert stage.lp_multiple_optima
    assert np.isclose(stage.lp_objective, 10.0)
    assert np.allclose(stage.lp_values_range, [[0, 2], [0, 2], [0, 0], [0, 0]])
    assert result.e_.stages[1].lp_values_range is None  # proven unique


def test_SIMUS_unbounded_stage_reported():
    # maximize C0 subject only to a ">=" constraint: unbounded
    dm = skcriteria.mkdm(
        matrix=[[1, 2], [3, 1], [2, 2]],
        objectives=[max, min],
        criteria=["C0", "C1"],
    )
    with pytest.warns(UserWarning, match=r"\('C0', 'Unbounded'\)"):
        result = SIMUS().evaluate(dm)

    stage = result.e_.stages[0]
    assert stage.lp_status == "Unbounded"
    assert stage.lp_status_code == 3
    assert stage.lp_objective is None
    assert np.all(stage.lp_values == 0)
    assert np.all(result.e_.stages_results[0] == 0)


def test_SIMUS_single_criterion_no_constraints():
    # min 0*x0 + 1*x1 + 2*x2, x >= 0 (no constraints): z* = 0 and x0 is
    # free on the optimal face
    dm = skcriteria.mkdm(matrix=[[0], [1], [2]], objectives=[min])
    with pytest.warns(UserWarning, match="multiple"):
        result = SIMUS().evaluate(dm)

    stage = result.e_.stages[0]
    assert stage.lp_status == "Optimal"
    assert stage.lp_objective == 0
    assert stage.lp_problem.A_ub.shape == (0, 3)
    assert stage.lp_values_range[0, 1] == np.inf
    assert np.all(stage.lp_values_range[1:] == 0)
