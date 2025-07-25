#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.cmp.rrtest1"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg.topsis import TOPSIS
from skcriteria.ranksrev.rank_invariant_check import RankInvariantChecker
from skcriteria.utils import rank


# =============================================================================
# TESTS
# =============================================================================


def test_RankInvariantChecker_decision_maker_no_evaluate_method():
    class NoEvaluateMethod:
        pass

    dmaker = NoEvaluateMethod()
    with pytest.raises(TypeError):
        RankInvariantChecker(dmaker)


def test_RankInvariantChecker_decision_maker_evaluate_no_callable():
    class EvaluateNoCallable:
        evaluate = None

    dmaker = EvaluateNoCallable()
    with pytest.raises(TypeError):
        RankInvariantChecker(dmaker)


def test_RankInvariantChecker_invalid_last_diff_strategy():
    class FakeDM:
        def evaluate(self): ...

    dmaker = FakeDM()
    with pytest.raises(TypeError):
        RankInvariantChecker(dmaker, last_diff_strategy=None)


# CHECK DOMINANCE =============================================================


def original_dominates_mutated(dm, result, alt_name):
    original = dm.alternatives[alt_name]
    noise = result[f"M.{alt_name}"].e_.rrt1.noise
    mutated = original + noise

    dom = rank.dominance(original, mutated, dm.minwhere)
    return dom.aDb > dom.bDa


def test_RankInvariantChecker_simple_stock_selection():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = TOPSIS()
    rrt1 = RankInvariantChecker(dmaker, random_state=42)
    result = rrt1.evaluate(dm)

    assert original_dominates_mutated(dm, result, "AA")
    assert original_dominates_mutated(dm, result, "MM")
    assert original_dominates_mutated(dm, result, "PE")
    assert original_dominates_mutated(dm, result, "JN")
    assert original_dominates_mutated(dm, result, "FX")


@pytest.mark.parametrize("windows_size", [7, 15])
def test_RankInvariantChecker_van2021evaluation(windows_size):
    dm = skc.datasets.load_van2021evaluation(windows_size=windows_size)
    dmaker = TOPSIS()
    rrt1 = RankInvariantChecker(dmaker, random_state=42)
    result = rrt1.evaluate(dm)

    assert original_dominates_mutated(dm, result, "ETH")
    assert original_dominates_mutated(dm, result, "LTC")
    assert original_dominates_mutated(dm, result, "XLM")
    assert original_dominates_mutated(dm, result, "BNB")
    assert original_dominates_mutated(dm, result, "ADA")
    assert original_dominates_mutated(dm, result, "LINK")
    assert original_dominates_mutated(dm, result, "XRP")
    assert original_dominates_mutated(dm, result, "DOGE")


# REMOVE AN ALTERNATIVE =======================================================


class RemoveAlternativeDMaker:
    def __init__(self, dmaker, remove, remove_at_call):
        self.dmaker = dmaker
        self.remove = remove
        self.remove_at_call = remove_at_call
        self._call_cnt = 0

    def filter_dm(self, dm, remove):
        mtx = dm.matrix
        filtered = mtx.loc[~mtx.index.isin(remove)]
        filtered_dm = dm.replace(
            matrix=filtered.to_numpy(),
            alternatives=filtered.index.to_numpy(),
        )
        return filtered_dm

    def evaluate(self, dm):
        dm = (
            self.filter_dm(dm, self.remove)
            if self._call_cnt == self.remove_at_call
            else dm
        )
        rank = self.dmaker.evaluate(dm)
        self._call_cnt += 1
        return rank


def test_RankInvariantChecker_remove_one_alternative_forbidden():
    dm = skc.datasets.load_simple_stock_selection()

    dmaker = RemoveAlternativeDMaker(TOPSIS(), ["AA"], 1)
    rrt1 = RankInvariantChecker(
        dmaker, random_state=42, allow_missing_alternatives=False
    )

    with pytest.raises(ValueError):
        rrt1.evaluate(dm)


def test_RankInvariantChecker_remove_one_alternative():
    dm = skc.datasets.load_simple_stock_selection()

    dmaker = RemoveAlternativeDMaker(TOPSIS(), ["AA"], 1)
    rrt1 = RankInvariantChecker(
        dmaker, random_state=42, allow_missing_alternatives=True
    )

    result = rrt1.evaluate(dm)

    _, rank = result.ranks[1]

    np.testing.assert_array_equal(rank.e_.rrt1.missing_alternatives, ["AA"])
    assert rank.to_series()["AA"] == 6


def test_RankInvariantChecker_remove_two_alternatives():
    dm = skc.datasets.load_simple_stock_selection()

    dmaker = RemoveAlternativeDMaker(TOPSIS(), ["AA", "MM"], 1)
    rrt1 = RankInvariantChecker(
        dmaker, random_state=42, allow_missing_alternatives=True
    )

    result = rrt1.evaluate(dm)

    _, rank = result.ranks[1]

    np.testing.assert_array_equal(
        rank.e_.rrt1.missing_alternatives, ["AA", "MM"]
    )

    assert rank.to_series()["AA"] == 5
    assert rank.to_series()["MM"] == 5
    assert rank.has_ties_


def test_RankInvariantChecker_repr():
    dmaker = TOPSIS()
    rrt1 = RankInvariantChecker(dmaker, random_state=42)

    result = repr(rrt1)
    expected = (
        f"<RankInvariantChecker {dmaker!r} repeats={1}, "
        f"allow_missing_alternatives={False} "
        f"last_diff_strategy={np.median!r}>"
    )

    assert result == expected
