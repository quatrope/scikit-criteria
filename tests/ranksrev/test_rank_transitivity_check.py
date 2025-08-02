#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Tests for the functionalities in the tranistivity_check file
"""


# =============================================================================
# IMPORTS
# =============================================================================

import networkx as nx

import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg.electre import ELECTRE2
from skcriteria.agg.moora import ReferencePointMOORA
from skcriteria.agg.topsis import TOPSIS
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing.filters import FilterNonDominated
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler, VectorScaler
from skcriteria.ranksrev.rank_transitivity_check import (
    RankTransitivityChecker,
    _format_transitivity_cycles,
    _transitivity_break_bound,
)
from skcriteria.utils.cycle_removal import (
    _select_edge_random,
    _select_edge_weighted,
)

# =============================================================================
# SHARED OBJECTS
# =============================================================================

# Pipeline to apply to all pairwise sub-problems
electre2_pipe = mkpipe(
    InvertMinimize(),
    FilterNonDominated(),
    SumScaler(target="weights"),
    VectorScaler(target="matrix"),
    ELECTRE2(),
)

topsis_pipe = mkpipe(
    InvertMinimize(),
    FilterNonDominated(),
    TOPSIS(),
)

topsis_pipe_matrix_scaler = mkpipe(
    InvertMinimize(),
    FilterNonDominated(),
    VectorScaler(target="matrix"),
    TOPSIS(),
)

topsis_pipe_moora = mkpipe(
    InvertMinimize(),
    FilterNonDominated(),
    VectorScaler(target="matrix"),
    ReferencePointMOORA(),
)

# =============================================================================
# STATIC FUNCTIONS
# =============================================================================


def test_TransitivityCheck_transitivity_break_bound_even():
    value = 10
    expected = 40
    actual = _transitivity_break_bound(value)
    assert actual == expected


def test_TransitivityCheck_transitivity_break_bound_odd():
    value = 11
    expected = 55
    actual = _transitivity_break_bound(value)
    assert actual == expected


def test_TransitivityCheck_format_transitivity_cycles_no_transitivity_break():
    dm = skc.datasets.load_simple_stock_selection()
    orank = electre2_pipe.evaluate(dm)
    trans_checker = RankTransitivityChecker(electre2_pipe)
    graph = trans_checker._dominance_graph(dm, orank)
    trans_break = list(nx.simple_cycles(graph, length_bound=3))
    result = _format_transitivity_cycles(trans_break)
    assert result == []


def test_TransitivityCheck_format_transitivity_cycles_transitivity_break():
    dm = skc.datasets.load_van2021evaluation(windows_size=7)
    orank = topsis_pipe_moora.evaluate(dm)
    trans_checker = RankTransitivityChecker(topsis_pipe_moora)
    graph = trans_checker._dominance_graph(dm, orank)
    trans_break = list(nx.simple_cycles(graph, length_bound=3))
    result = _format_transitivity_cycles(trans_break)
    assert result != []


# =============================================================================
# PROPERTIES
# =============================================================================


def test_TransitivityChecker_repr():
    trans_checker = RankTransitivityChecker(electre2_pipe)
    assert repr(trans_checker) == (
        f"<{trans_checker.get_method_name()} "
        f"{repr(trans_checker.dmaker)}, "
        f"cycle_removal_strategy="
        f"{trans_checker.cycle_removal_strategy}, "
        f"max_ranks={trans_checker.max_ranks}>"
    )


def test_TransitivityChecker_bad_pipe():
    bad_pipe = "Suffering and pain"
    with pytest.raises(TypeError) as ex:
        RankTransitivityChecker(bad_pipe)
        assert "'dmaker' must implement 'evaluate()' method" in str(ex.value)


def test_TransitivityChecker_dmaker():
    trans_checker = RankTransitivityChecker(electre2_pipe)
    assert trans_checker.dmaker == electre2_pipe


def test_TransitivityChecker_bad_fallback():
    bad_pipe = "Suffering and pain 2"
    with pytest.raises(TypeError) as ex:
        RankTransitivityChecker(topsis_pipe, fallback=bad_pipe)
        assert "'fallback' must implement 'evaluate()' method" in str(ex.value)


def test_TransitivityChecker_fallback():
    trans_checker = RankTransitivityChecker(topsis_pipe, fallback=topsis_pipe)
    assert trans_checker.fallback == topsis_pipe


def test_TransitivityChecker_parallell_backend_none():
    trans_checker = RankTransitivityChecker(electre2_pipe)
    assert trans_checker.parallel_backend is None


def test_TransitivityChecker_parallell_backend():
    trans_checker = RankTransitivityChecker(
        electre2_pipe, parallel_backend=electre2_pipe
    )
    assert trans_checker.parallel_backend == electre2_pipe


def test_TransitivityChecker_random_state():
    rnd_state = 42
    trans_checker = RankTransitivityChecker(
        electre2_pipe, random_state=rnd_state
    )
    assert (
        trans_checker.random_state.random()
        == np.random.default_rng(rnd_state).random()
    )


def test_TransitivityChecker_make_transitivity_strategy_random():
    trans_checker = RankTransitivityChecker(electre2_pipe)
    assert trans_checker.cycle_removal_strategy == _select_edge_random


def test_TransitivityChecker_make_transitivity_strategy_weighted():
    trans_checker = RankTransitivityChecker(
        electre2_pipe, cycle_removal_strategy="weighted"
    )

    assert trans_checker.cycle_removal_strategy == _select_edge_weighted


def test_TransitivityChecker_make_transitivity_strategy_divination():
    bad_strat = "Divination"
    with pytest.raises(ValueError):
        RankTransitivityChecker(
            electre2_pipe, cycle_removal_strategy=bad_strat
        )


def test_TransitivityChecker_allow_missing_alternatives_default():
    trans_checker = RankTransitivityChecker(topsis_pipe)
    assert trans_checker.allow_missing_alternatives is False


def test_TransitivityChecker_allow_missing_alternatives_True():
    trans_checker = RankTransitivityChecker(
        topsis_pipe, allow_missing_alternatives=True
    )
    assert trans_checker.allow_missing_alternatives is True


def test_TransitivityChecker_max_ranks_default():
    trans_checker = RankTransitivityChecker(electre2_pipe)
    assert trans_checker.max_ranks == 50


def test_TransitivityChecker_max_ranks_custom():
    ranks = 42
    trans_checker = RankTransitivityChecker(electre2_pipe, max_ranks=ranks)
    assert trans_checker.max_ranks == ranks


def test_TransitivityChecker_max_ranks_zero():
    ranks = 0
    with pytest.raises(ValueError):
        RankTransitivityChecker(electre2_pipe, max_ranks=ranks)


def test_TransitivityChecker_n_jobs_default():
    trans_checker = RankTransitivityChecker(electre2_pipe)
    assert trans_checker.n_jobs is None


def test_TransitivityChecker_n_jobs_custom():
    jobs = 42
    trans_checker = RankTransitivityChecker(electre2_pipe, n_jobs=jobs)
    assert trans_checker.n_jobs == jobs


# =============================================================================
# TEST MISSING ALTERNATIVES
# =============================================================================


def test_TransitivityCheck_missing_alternative_forbidden():
    dm = skc.datasets.load_simple_stock_selection()
    trans_check = RankTransitivityChecker(
        topsis_pipe, random_state=42, allow_missing_alternatives=False
    )
    with pytest.raises(ValueError):
        trans_check.evaluate(dm=dm)


def test_TransitivityCheck_missing_alternative():
    dm = skc.datasets.load_simple_stock_selection()
    trans_check = RankTransitivityChecker(
        topsis_pipe, random_state=42, allow_missing_alternatives=True
    )
    result = trans_check.evaluate(dm=dm)

    _, rank = result.ranks[1]

    np.testing.assert_array_equal(
        rank.e_.transitivity_check.missing_alternatives, ["FX", "MM"]
    )

    assert rank.to_series()["FX"] == 5
    assert rank.to_series()["MM"] == 5
    assert rank.has_ties_


# ============================================================================
# TEST CRITERIA
# =============================================================================


def test_TransitivityCheck_test_criterion_2_pass():
    dm = skc.datasets.load_van2021evaluation(windows_size=7)
    trans_check = RankTransitivityChecker(topsis_pipe)
    rank_comparator = trans_check.evaluate(dm=dm)
    orank = topsis_pipe.evaluate(dm)
    test_criterion_2 = trans_check._test_criterion_2(dm, orank)[0]
    assert rank_comparator._extra.transitivity_break_rate == 0
    assert rank_comparator._extra.test_criterion_2
    assert test_criterion_2


def test_TransitivityCheck_test_criterion_2_fail():
    dm = skc.datasets.load_van2021evaluation(windows_size=7)
    trans_check = RankTransitivityChecker(topsis_pipe_moora)
    rank_comparator = trans_check.evaluate(dm=dm)
    orank = topsis_pipe.evaluate(dm)
    test_criterion_2 = trans_check._test_criterion_2(dm, orank)[0]
    assert rank_comparator._extra.transitivity_break_rate > 0
    assert not rank_comparator._extra.test_criterion_2
    assert not test_criterion_2


def test_TransitivityCheck_test_criterion_3_pass():
    dm = skc.datasets.load_van2021evaluation(windows_size=7)
    trans_check = RankTransitivityChecker(topsis_pipe)
    rank_comparator = trans_check.evaluate(dm=dm)
    assert rank_comparator._extra.test_criterion_3


def test_TransitivityCheck_test_criterion_3_fail():
    dm = skc.datasets.load_van2021evaluation(windows_size=7)
    trans_check = RankTransitivityChecker(topsis_pipe_matrix_scaler)
    rank_comparator = trans_check.evaluate(dm=dm)
    assert not rank_comparator._extra.test_criterion_3
