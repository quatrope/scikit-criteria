#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.cmp.rrtest1

"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

import pandas as pd

import pytest

import skcriteria as skc
from skcriteria.madm.similarity import TOPSIS
from skcriteria.cmp.rankrev.rrtest1 import RankReversalTest1
from skcriteria.utils import rank

# =============================================================================
# TESTS
# =============================================================================


def test_RankReversalTest1_decision_maker_no_evaluate_method():
    class NoEvaluateMethod:
        ...

    dmaker = NoEvaluateMethod()
    with pytest.raises(TypeError):
        RankReversalTest1(dmaker)


def test_RankReversalTest1_decision_maker_evaluate_no_callable():
    class EvaluateNoCallable:
        evaluate = None

    dmaker = EvaluateNoCallable()
    with pytest.raises(TypeError):
        RankReversalTest1(dmaker)


def test_RankReversalTest1_invalid_last_diff_strategy():
    class FakeDM:
        def evaluate(self):
            ...

    dmaker = FakeDM()
    with pytest.raises(TypeError):
        RankReversalTest1(dmaker, last_diff_strategy=None)


def test_RankReversalTest1_simple_stock_selection():
    dm = skc.datasets.load_simple_stock_selection()
    dmaker = TOPSIS()
    rrt1 = RankReversalTest1(dmaker, seed=42)
    result = rrt1.evaluate(dm)

    def get_orinal_and_mutated(dm, result, alt_name):
        original = dm.alternatives[alt_name]
        noise = result[f"M.{alt_name}"].e_.rrt1.noise
        mutated = original + noise
        return original, mutated

    def original_dominates_mutated(dm, result, alt_name):
        original, mutated = get_orinal_and_mutated(dm, result, alt_name)
        dom = rank.dominance(original, mutated, dm.minwhere)
        return dom.aDb > dom.bDa

    assert original_dominates_mutated(dm, result, "AA")
    assert original_dominates_mutated(dm, result, "MM")
    assert original_dominates_mutated(dm, result, "PE")
    assert original_dominates_mutated(dm, result, "JN")
    assert original_dominates_mutated(dm, result, "FX")


@pytest.mark.parametrize("windows_size", [7, 15])
def test_RankReversalTest1_van2021evaluation(windows_size):
    dm = skc.datasets.load_van2021evaluation(windows_size=windows_size)
    dmaker = TOPSIS()
    rrt1 = RankReversalTest1(dmaker, seed=42)
    result = rrt1.evaluate(dm)

    def get_orinal_and_mutated(dm, result, alt_name):
        original = dm.alternatives[alt_name]
        noise = result[f"M.{alt_name}"].e_.rrt1.noise
        mutated = original + noise
        return original, mutated

    def original_dominates_mutated(dm, result, alt_name):
        original, mutated = get_orinal_and_mutated(dm, result, alt_name)
        dom = rank.dominance(original, mutated, dm.minwhere)
        return dom.aDb > dom.bDa

    assert original_dominates_mutated(dm, result, "ETH")
    assert original_dominates_mutated(dm, result, "LTC")
    assert original_dominates_mutated(dm, result, "XLM")
    assert original_dominates_mutated(dm, result, "BNB")
    assert original_dominates_mutated(dm, result, "ADA")
    assert original_dominates_mutated(dm, result, "LINK")
    assert original_dominates_mutated(dm, result, "XRP")
    assert original_dominates_mutated(dm, result, "DOGE")
