#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.lp"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria.utils import lp

# =============================================================================
# tests
# =============================================================================


@pytest.mark.parametrize(
    "solver, expected",
    [
        (None, True),
        ("PULP_CBC_CMD", True),
        ("pulp", True),
        ("PuLP", True),
        ("PULP", True),
        ("foo", False),
    ],
)
def test_is_available(solver, expected):
    assert lp.is_solver_available(solver) is expected


@pytest.mark.parametrize(
    "solver", [None, "PULP_CBC_CMD", "pulp", "PuLP", "PULP"]
)
def test_maximize(solver):
    x0 = lp.Float("x0", low=0)
    x1 = lp.Float("x1", low=0)
    x2 = lp.Float("x2", low=0)

    model = lp.Maximize(
        z=250 * x0 + 130 * x1 + 350 * x2, solver=solver
    ).subject_to(
        120 * x0 + 200 * x1 + 340 * x2 <= 500,
        -20 * x0 + -40 * x1 + -15 * x2 <= -15,
        800 * x0 + 1000 * x1 + 600 * x2 <= 1000,
    )

    assert x0 is model.v.x0
    assert x1 is model.v.x1
    assert x2 is model.v.x2

    result = model.solve()
    assert result.lp_status == "Optimal"
    assert result.lp_objective == 540
    assert np.all(result.lp_variables == ["x0", "x1", "x2"])
    assert np.all(result.lp_values == [0.2, 0.0, 1.4])

    expected_repr = (
        "Maximize(250*x0 + 130*x1 + 350*x2).subject_to(\n"
        "  120*x0 + 200*x1 + 340*x2 <= 500,\n"
        "  -20*x0 - 40*x1 - 15*x2 <= -15,\n"
        "  800*x0 + 1000*x1 + 600*x2 <= 1000\n"
        ")"
    )

    assert repr(model) == expected_repr


@pytest.mark.parametrize(
    "solver", [None, "PULP_CBC_CMD", "pulp", "PuLP", "PULP"]
)
def test_minimize_from_matrix(solver):
    x0 = lp.Float("x0", low=0)
    x1 = lp.Float("x1", low=0)
    x2 = lp.Float("x2", low=0)

    model = lp.Minimize(
        z=250 * x0 + 130 * x1 + 350 * x2, solver=solver
    ).subject_to(
        120 * x0 + 200 * x1 + 340 * x2 >= 500,
        20 * x0 + 40 * x1 + 15 * x2 >= 15,
        800 * x0 + 1000 * x1 + 600 * x2 >= 1000,
    )

    assert x0 is model.v.x0
    assert x1 is model.v.x1
    assert x2 is model.v.x2

    result = model.solve()
    assert result.lp_status == "Optimal"
    assert result.lp_objective == 325
    assert np.all(result.lp_variables == ["x0", "x1", "x2"])
    assert np.all(result.lp_values == [0.0, 2.5, 0.0])

    expected_repr = (
        "Minimize(250*x0 + 130*x1 + 350*x2).subject_to(\n"
        "  120*x0 + 200*x1 + 340*x2 >= 500,\n"
        "  20*x0 + 40*x1 + 15*x2 >= 15,\n"
        "  800*x0 + 1000*x1 + 600*x2 >= 1000\n"
        ")"
    )

    assert repr(model) == expected_repr
