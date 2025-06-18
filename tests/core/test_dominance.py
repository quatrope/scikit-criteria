#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.core.dominance"""


# =============================================================================
# IMPORTS
# =============================================================================

import inspect
from unittest import mock

import numpy as np

import pandas as pd

import pytest

import skcriteria as skc
from skcriteria.core import data, dominance

# =============================================================================
# TEST IF __call__ calls the correct method
# =============================================================================


def test_DecisionMatrixDominanceAccessor_call_invalid_kind(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=3,
        max_alternatives=3,
        min_criteria=3,
        max_criteria=3,
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    with pytest.raises(ValueError):
        dom("__call__")

    dom.zaraza = None  # not callable
    with pytest.raises(ValueError):
        dom("zaraza")


@pytest.mark.parametrize(
    "kind",
    {
        kind
        for kind, kind_method in vars(
            dominance.DecisionMatrixDominanceAccessor
        ).items()
        if not inspect.ismethod(kind_method) and not kind.startswith("_")
    },
)
def test_DecisionMatrixDominanceAccessor_call(decision_matrix, kind):
    dm = decision_matrix(
        seed=42,
        min_alternatives=3,
        max_alternatives=3,
        min_criteria=3,
        max_criteria=3,
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    method_name = (
        f"skcriteria.core.dominance.DecisionMatrixDominanceAccessor.{kind}"
    )

    with mock.patch(method_name) as plot_method:
        dom(kind=kind)

    plot_method.assert_called_once()


# =============================================================================
# BT
# =============================================================================


def test_DecisionMatrixDominanceAccessor_bt():
    dm = data.mkdm(
        matrix=[
            [10, 40],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.DataFrame(
        [
            [0, 1],
            [1, 0],
        ],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )
    expected.index.name = "Better than"
    expected.columns.name = "Worse than"

    pd.testing.assert_frame_equal(dom.bt(), expected)
    pd.testing.assert_frame_equal(dm.dominance.bt(), expected)


# =============================================================================
# EQ
# =============================================================================


def test_DecisionMatrixDominanceAccessor_eq():
    dm = data.mkdm(
        matrix=[
            [10, 70],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.DataFrame(
        [
            [2, 1],
            [1, 2],
        ],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )
    expected.index.name = "Equals to"
    expected.columns.name = "Equals to"

    pd.testing.assert_frame_equal(dom.eq(), expected)
    pd.testing.assert_frame_equal(dm.dominance.eq(), expected)


def test_DecisionMatrixDominanceAccessor_eq_simple_stock_selection():
    dm = skc.datasets.load_simple_stock_selection()
    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.DataFrame(
        [
            [3, 0, 0, 0, 0, 0],
            [0, 3, 1, 1, 0, 1],
            [0, 1, 3, 0, 0, 1],
            [0, 1, 0, 3, 0, 0],
            [0, 0, 0, 0, 3, 1],
            [0, 1, 1, 0, 1, 3],
        ],
        index=["PE", "JN", "AA", "FX", "MM", "GN"],
        columns=["PE", "JN", "AA", "FX", "MM", "GN"],
    )
    expected.index.name = "Equals to"
    expected.columns.name = "Equals to"

    pd.testing.assert_frame_equal(dom.eq(), expected)
    pd.testing.assert_frame_equal(dm.dominance.eq(), expected)


# =============================================================================
# COMPARE
# =============================================================================


def test_DecisionMatrixDominanceAccessor_compare():
    dm = data.mkdm(
        matrix=[
            [10, 70],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.DataFrame.from_dict(
        {
            ("Criteria", "C0"): {
                ("Alternatives", "A0"): False,
                ("Alternatives", "A1"): True,
                ("Equals", ""): False,
            },
            ("Criteria", "C1"): {
                ("Alternatives", "A0"): False,
                ("Alternatives", "A1"): False,
                ("Equals", ""): True,
            },
            ("Performance", ""): {
                ("Alternatives", "A0"): 0,
                ("Alternatives", "A1"): 1,
                ("Equals", ""): 1,
            },
        }
    )

    result = dom.compare("A0", "A1")

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


# =============================================================================
# DOMINANCE
# =============================================================================


def test_DecisionMatrixDominanceAccessor_dominance():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.DataFrame(
        [
            [False, False],
            [True, False],
        ],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )
    expected.index.name = "Dominators"
    expected.columns.name = "Dominated"

    pd.testing.assert_frame_equal(dom.dominance(), expected)
    pd.testing.assert_frame_equal(dm.dominance.dominance(), expected)


def test_DecisionMatrixDominanceAccessor_dominance_strict():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.DataFrame(
        [
            [False, False],
            [True, False],
        ],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )
    expected.index.name = "Strict dominators"
    expected.columns.name = "Strictly dominated"

    pd.testing.assert_frame_equal(dom.dominance(strict=True), expected)
    pd.testing.assert_frame_equal(
        dm.dominance.dominance(strict=True), expected
    )


def test_DecisionMatrixDominanceAccessor_dominance_strict_false():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [10, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    strict_expected = pd.DataFrame(
        [
            [False, False],
            [False, False],
        ],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )
    strict_expected.index.name = "Strict dominators"
    strict_expected.columns.name = "Strictly dominated"

    pd.testing.assert_frame_equal(dom.dominance(strict=True), strict_expected)
    pd.testing.assert_frame_equal(
        dm.dominance.dominance(strict=True), strict_expected
    )

    not_strict_expected = pd.DataFrame(
        [
            [False, False],
            [True, False],
        ],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )
    not_strict_expected.index.name = "Dominators"
    not_strict_expected.columns.name = "Dominated"

    pd.testing.assert_frame_equal(
        dom.dominance(strict=False), not_strict_expected
    )
    pd.testing.assert_frame_equal(
        dm.dominance.dominance(strict=False), not_strict_expected
    )


# =============================================================================
# DOMINATED
# =============================================================================


def test_DecisionMatrixDominanceAccessor_dominated():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.Series([True, False], index=["A0", "A1"], name="Dominated")
    expected.index.name = "Alternatives"

    pd.testing.assert_series_equal(dom.dominated(), expected)
    pd.testing.assert_series_equal(dm.dominance.dominated(), expected)


def test_DecisionMatrixDominanceAccessor_dominated_strict():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    expected = pd.Series(
        [True, False], index=["A0", "A1"], name="Strictly dominated"
    )
    expected.index.name = "Alternatives"

    pd.testing.assert_series_equal(dom.dominated(strict=True), expected)
    pd.testing.assert_series_equal(
        dm.dominance.dominated(strict=True), expected
    )


def test_DecisionMatrixDominanceAccessor_dominated_strict_false():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [10, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    strict_expected = pd.Series(
        [False, False], index=["A0", "A1"], name="Strictly dominated"
    )
    strict_expected.index.name = "Alternatives"

    pd.testing.assert_series_equal(dom.dominated(strict=True), strict_expected)
    pd.testing.assert_series_equal(
        dm.dominance.dominated(strict=True), strict_expected
    )

    not_strict_expected = pd.Series(
        [True, False], index=["A0", "A1"], name="Dominated"
    )
    not_strict_expected.index.name = "Alternatives"

    pd.testing.assert_series_equal(
        dom.dominated(strict=False), not_strict_expected
    )
    pd.testing.assert_series_equal(
        dm.dominance.dominated(strict=False), not_strict_expected
    )


# =============================================================================
# DOMINATORS OF
# =============================================================================


def test_DecisionMatrixDominanceAccessor_dominators_of():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [20, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    assert np.all(dom.dominators_of("A0") == ["A1"])
    assert np.all(dm.dominance.dominators_of("A0") == ["A1"])

    assert not len(dom.dominators_of("A1"))
    assert not len(dm.dominance.dominators_of("A1"))


def test_DecisionMatrixDominanceAccessor_dominators_of_strict():
    dm = data.mkdm(
        matrix=[
            [20, 80],
            [20, 90],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    assert not len(dom.dominators_of("A1", strict=True))
    assert not len(dm.dominance.dominators_of("A1", strict=True))

    assert not len(dom.dominators_of("A0", strict=True))
    assert not len(dm.dominance.dominators_of("A0", strict=True))


def test_DecisionMatrixDominanceAccessor_dominators_of_strict_false():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [10, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    assert np.all(dom.dominators_of("A0", strict=False) == ["A1"])
    assert not len(dom.dominators_of("A0", strict=True))


# =============================================================================
# HAS LOOPS
# =============================================================================


def test_DecisionMatrixDominanceAccessor_has_loops_false():
    dm = data.mkdm(
        matrix=[
            [10, 80],
            [10, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    dom = dominance.DecisionMatrixDominanceAccessor(dm)

    assert not dom.has_loops(strict=True)
    assert not dom.has_loops(strict=False)


def test_DecisionMatrixDominanceAccessor_has_loops_true():
    """This test is complex so we relay on a hack"""

    dm = data.mkdm(
        matrix=[
            [10, 80],
            [10, 70],
        ],
        objectives=[max, min],
        alternatives=["A0", "A1"],
        criteria=["C0", "C1"],
    )

    fake_dominance = pd.DataFrame(
        [[False, True], [True, False]],
        index=["A0", "A1"],
        columns=["A0", "A1"],
    )

    with mock.patch.object(
        dm.dominance, "dominance", return_value=fake_dominance
    ):
        assert dm.dominance.has_loops()
