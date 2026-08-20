#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.importance.criteria_only_one"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

import pytest

import skcriteria as skc
from skcriteria import mkdm
from skcriteria.agg import simple
from skcriteria.agg.topsis import TOPSIS
from skcriteria.importance import CriteriaOnlyOneChecker

# =============================================================================
# HELPERS
# =============================================================================


class DropAlternativeDMaker:
    """Wraps a dmaker and always drops ``drop`` from any dm it evaluates."""

    def __init__(self, dmaker, drop):
        self.dmaker = dmaker
        self.drop = drop

    def evaluate(self, dm):
        if self.drop in dm.alternatives:
            mtx = dm.matrix
            filtered = mtx.loc[~mtx.index.isin([self.drop])]
            dm = dm.replace(
                matrix=filtered.to_numpy(),
                alternatives=filtered.index.to_numpy(),
            )
        return self.dmaker.evaluate(dm)


def _reversal_dm():
    """2-criteria, 3-alternative dm crafted so that keeping only ``C0``
    reproduces the reference ranking exactly, and keeping only ``C1``
    fully reverses it.

    """
    return mkdm(
        matrix=[[1, 3], [2, 2], [3, 1]],
        objectives=[max, max],
        weights=[2, 1],
        alternatives=["A", "B", "C"],
        criteria=["C0", "C1"],
    )


# =============================================================================
# CONSTRUCTOR VALIDATION
# =============================================================================


def test_CriteriaOnlyOneChecker_dmaker_no_evaluate_method():
    class NoEvaluateMethod:
        pass

    with pytest.raises(TypeError):
        CriteriaOnlyOneChecker(NoEvaluateMethod())


def test_CriteriaOnlyOneChecker_dmaker_evaluate_not_callable():
    class EvaluateNotCallable:
        evaluate = None

    with pytest.raises(TypeError):
        CriteriaOnlyOneChecker(EvaluateNotCallable())


def test_CriteriaOnlyOneChecker_invalid_metric():
    with pytest.raises(ValueError):
        CriteriaOnlyOneChecker(simple.WeightedSumModel(), metric="pearson")


# =============================================================================
# EVALUATE VALIDATION
# =============================================================================


def test_CriteriaOnlyOneChecker_requires_at_least_two_criteria():
    dm = mkdm(
        matrix=[[1], [2], [3]],
        objectives=[max],
        alternatives=["A", "B", "C"],
        criteria=["C0"],
    )
    checker = CriteriaOnlyOneChecker(simple.WeightedSumModel())
    with pytest.raises(ValueError):
        checker.evaluate(dm)


# =============================================================================
# STRUCTURE OF THE RESULT
# =============================================================================


def test_CriteriaOnlyOneChecker_ranks_names():
    dm = _reversal_dm()
    checker = CriteriaOnlyOneChecker(simple.WeightedSumModel())
    result = checker.evaluate(dm)

    assert [name for name, _ in result.ranks] == [
        "reference",
        "OO-C0",
        "OO-C1",
    ]

    importance = result.extra_["importance"]
    assert set(importance.index) == {"C0", "C1"}


# =============================================================================
# HAND-COMPUTED IMPORTANCE VALUES
# =============================================================================


@pytest.mark.parametrize("metric", ["footrule", "kendall"])
def test_CriteriaOnlyOneChecker_importance_values(metric):
    dm = _reversal_dm()
    checker = CriteriaOnlyOneChecker(simple.WeightedSumModel(), metric=metric)
    result = checker.evaluate(dm)

    importance = result.extra_["importance"]

    # C0 alone reproduces the reference ranking exactly -> maximum
    # importance (it is sufficient on its own)
    assert importance["C0"] == pytest.approx(1.0)

    # C1 alone fully reverses the reference ranking -> zero importance
    assert importance["C1"] == pytest.approx(0.0)


@pytest.mark.parametrize("metric", ["footrule", "kendall"])
def test_CriteriaOnlyOneChecker_importance_bounded(metric):
    dm = skc.datasets.load_simple_stock_selection()
    checker = CriteriaOnlyOneChecker(TOPSIS(), metric=metric)
    importance = checker.evaluate(dm).extra_["importance"]

    assert (importance >= 0.0).all()
    assert (importance <= 1.0).all()


# =============================================================================
# MISSING ALTERNATIVES
# =============================================================================


def test_CriteriaOnlyOneChecker_missing_alternative_forbidden():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = DropAlternativeDMaker(TOPSIS(), "AA")
    checker = CriteriaOnlyOneChecker(dmaker, allow_missing_alternatives=False)

    with pytest.raises(ValueError, match="AA"):
        checker.evaluate(dm)


def test_CriteriaOnlyOneChecker_missing_alternative_allowed():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = DropAlternativeDMaker(TOPSIS(), "AA")
    checker = CriteriaOnlyOneChecker(dmaker, allow_missing_alternatives=True)

    result = checker.evaluate(dm)

    for _, rank in result.ranks:
        assert "AA" in rank.alternatives


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================


def test_CriteriaOnlyOneChecker_parallel_matches_sequential():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = TOPSIS()

    sequential = CriteriaOnlyOneChecker(dmaker).evaluate(dm)
    parallel = CriteriaOnlyOneChecker(
        dmaker, n_jobs=2, preferred_parallel_backend="threads"
    ).evaluate(dm)

    pd.testing.assert_series_equal(
        sequential.extra_["importance"], parallel.extra_["importance"]
    )


# =============================================================================
# REPR
# =============================================================================


def test_CriteriaOnlyOneChecker_repr():
    dmaker = simple.WeightedSumModel()
    checker = CriteriaOnlyOneChecker(dmaker)

    result = repr(checker)
    expected = (
        f"<CriteriaOnlyOneChecker [allow_missing_alternatives={True}, "
        f"dmaker={dmaker!r}, metric={'footrule'!r}, n_jobs={None}, "
        f"preferred_parallel_backend={None}, untied={False}]>"
    )

    assert result == expected
