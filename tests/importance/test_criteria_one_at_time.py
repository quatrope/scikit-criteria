#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.importance.criteria_one_at_time"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

import pytest

import skcriteria as skc
from skcriteria import mkdm
from skcriteria.agg import simple
from skcriteria.agg.topsis import TOPSIS
from skcriteria.importance import CriteriaOneAtATimeChecker

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
    """2-criteria, 3-alternative dm crafted so that, with ``delta=0.5``,
    lowering ``C0``'s weight fully reverses the ranking.

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


def test_CriteriaOneAtATimeChecker_dmaker_no_evaluate_method():
    class NoEvaluateMethod:
        pass

    with pytest.raises(TypeError):
        CriteriaOneAtATimeChecker(NoEvaluateMethod())


def test_CriteriaOneAtATimeChecker_dmaker_evaluate_not_callable():
    class EvaluateNotCallable:
        evaluate = None

    with pytest.raises(TypeError):
        CriteriaOneAtATimeChecker(EvaluateNotCallable())


def test_CriteriaOneAtATimeChecker_invalid_metric():
    with pytest.raises(ValueError):
        CriteriaOneAtATimeChecker(simple.WeightedSumModel(), metric="pearson")


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.1])
def test_CriteriaOneAtATimeChecker_invalid_delta(delta):
    with pytest.raises(ValueError):
        CriteriaOneAtATimeChecker(simple.WeightedSumModel(), delta=delta)


def test_CriteriaOneAtATimeChecker_delta_property():
    checker = CriteriaOneAtATimeChecker(simple.WeightedSumModel(), delta=0.3)
    assert checker.delta == pytest.approx(0.3)


# =============================================================================
# EVALUATE VALIDATION
# =============================================================================


def test_CriteriaOneAtATimeChecker_requires_at_least_two_criteria():
    dm = mkdm(
        matrix=[[1], [2], [3]],
        objectives=[max],
        alternatives=["A", "B", "C"],
        criteria=["C0"],
    )
    checker = CriteriaOneAtATimeChecker(simple.WeightedSumModel())
    with pytest.raises(ValueError):
        checker.evaluate(dm)


# =============================================================================
# STRUCTURE OF THE RESULT
# =============================================================================


def test_CriteriaOneAtATimeChecker_ranks_names():
    dm = _reversal_dm()
    checker = CriteriaOneAtATimeChecker(simple.WeightedSumModel())
    result = checker.evaluate(dm)

    assert [name for name, _ in result.ranks] == [
        "reference",
        "OAT(C0+0.2)",
        "OAT(C0-0.2)",
        "OAT(C1+0.2)",
        "OAT(C1-0.2)",
    ]

    importance = result.extra_["importance"]
    assert set(importance.index) == {"C0", "C1"}


# =============================================================================
# HAND-COMPUTED IMPORTANCE VALUES
# =============================================================================


def test_CriteriaOneAtATimeChecker_importance_values():
    dm = _reversal_dm()
    checker = CriteriaOneAtATimeChecker(simple.WeightedSumModel(), delta=0.5)
    result = checker.evaluate(dm)

    importance = result.extra_["importance"]

    # lowering C0's weight by delta=0.5 fully reverses the ranking, and
    # that's the worst of its two directions -> maximum importance
    assert importance["C0"] == pytest.approx(1.0)

    # C1's worst direction only partially disturbs the ranking
    assert importance["C1"] == pytest.approx(0.75)


@pytest.mark.parametrize("metric", ["footrule", "kendall"])
def test_CriteriaOneAtATimeChecker_importance_bounded(metric):
    dm = skc.datasets.load_simple_stock_selection()
    checker = CriteriaOneAtATimeChecker(TOPSIS(), metric=metric)
    importance = checker.evaluate(dm).extra_["importance"]

    assert (importance >= 0.0).all()
    assert (importance <= 1.0).all()


# =============================================================================
# MISSING ALTERNATIVES
# =============================================================================


def test_CriteriaOneAtATimeChecker_missing_alternative_forbidden():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = DropAlternativeDMaker(TOPSIS(), "AA")
    checker = CriteriaOneAtATimeChecker(dmaker, allow_missing_alternatives=False)

    with pytest.raises(ValueError, match="AA"):
        checker.evaluate(dm)


def test_CriteriaOneAtATimeChecker_missing_alternative_allowed():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = DropAlternativeDMaker(TOPSIS(), "AA")
    checker = CriteriaOneAtATimeChecker(dmaker, allow_missing_alternatives=True)

    result = checker.evaluate(dm)

    for _, rank in result.ranks:
        assert "AA" in rank.alternatives


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================


def test_CriteriaOneAtATimeChecker_parallel_matches_sequential():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = TOPSIS()

    sequential = CriteriaOneAtATimeChecker(dmaker).evaluate(dm)
    parallel = CriteriaOneAtATimeChecker(
        dmaker, n_jobs=2, preferred_parallel_backend="threads"
    ).evaluate(dm)

    pd.testing.assert_series_equal(
        sequential.extra_["importance"], parallel.extra_["importance"]
    )


# =============================================================================
# REPR
# =============================================================================


def test_CriteriaOneAtATimeChecker_repr():
    dmaker = simple.WeightedSumModel()
    checker = CriteriaOneAtATimeChecker(dmaker)

    result = repr(checker)
    expected = (
        f"<CriteriaOneAtATimeChecker [allow_missing_alternatives={True}, "
        f"delta={0.2!r}, dmaker={dmaker!r}, metric={'footrule'!r}, "
        f"n_jobs={None}, preferred_parallel_backend={None}, "
        f"untied={False}]>"
    )

    assert result == expected
