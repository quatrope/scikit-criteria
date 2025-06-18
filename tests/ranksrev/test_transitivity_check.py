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


import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg.electre import ELECTRE2
from skcriteria.agg.similarity import TOPSIS
from skcriteria.pipeline import mkpipe
from skcriteria.preprocessing.filters import FilterNonDominated
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler, VectorScaler
from skcriteria.ranksrev.rank_invariant_check import RankInvariantChecker
import skcriteria.ranksrev.transitivity_check
from skcriteria.utils import rank
import networkx as nx

# =============================================================================
# SHARED OBJECTS
# =============================================================================
# Pipeline to apply to all pairwise sub-problems
ws_pipe = mkpipe(
    InvertMinimize(),
    FilterNonDominated(),
    SumScaler(target="weights"),
    VectorScaler(target="matrix"),
    ELECTRE2(),
)

# =============================================================================
# TESTS
# =============================================================================

# =============================================================================
# STATIC FUNCTIONS
# =============================================================================
def test_TransitivityCheck_transitivity_break_bound_even():
    value = 10
    expected = 40
    actual =  skcriteria.ranksrev.transitivity_check._transitivity_break_bound(value)
    assert actual == expected

def test_TransitivityCheck_transitivity_break_bound_odd():
    value = 11
    expected = 55
    actual =  skcriteria.ranksrev.transitivity_check._transitivity_break_bound(value)
    assert actual == expected

def test_TransitivityCheck_untie_first():
    first = np.array([1,2,3,4,5])
    second = np.array([0,0,0,0,0])
    actual = skcriteria.ranksrev.transitivity_check._untie_first(first,second)
    assert actual == [(first, second)]

def test_TransitivityCheck_untie_second():
    first = np.array([1,2,3,4,5])
    second = np.array([0,0,0,0,0])
    actual = skcriteria.ranksrev.transitivity_check._untie_second(first,second)
    assert actual == [(second,first)]

def test_TransitivityCheck_untie_dominance_first():
    first = np.array([1,2,3,4,3])
    second = np.array([1,2,3,4,4])
    actual = skcriteria.ranksrev.transitivity_check._untie_by_dominance(first,second)
    assert actual == [(first,second)]

def test_TransitivityCheck_untie_dominance_second():
    first = np.array([42,0,0,42,42])
    second = np.array([1,2,3,4,4])
    actual = skcriteria.ranksrev.transitivity_check._untie_by_dominance(first,second)
    assert actual == [(second,first)]

def test_TransitivityCheck_in_degree_sort():
    dm = skc.datasets.load_simple_stock_selection()
    orank = ws_pipe.evaluate(dm)
    trans_checker = skcriteria.ranksrev.TransitivityChecker(ws_pipe)
    graph = trans_checker._dominance_graph(dm, orank)
    result = skcriteria.ranksrev.transitivity_check.in_degree_sort(graph)
    assert result == [['AA'], ['GN'], ['JN'], ['PE']]

def test_TransitivityCheck_assign_rankings():
    groups = [['AA'], ['GN'], ['JN'], ['PE']]
    result = skcriteria.ranksrev.transitivity_check.assign_rankings(groups)
    assert result == {'AA': 1, 'GN': 2, 'JN': 3, 'PE': 4}

def test_TransitivityCheck_format_transitivity_cycles_no_transitivity_break():
    dm = skc.datasets.load_non_rank_reversal_matrix()
    orank = ws_pipe.evaluate(dm)
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    graph = trans_checker._dominance_graph(dm, orank)
    trans_break = list(nx.simple_cycles(graph, length_bound=3))
    result = skcriteria.ranksrev.transitivity_check._format_transitivity_cycles(trans_break)
    assert result == []

def test_TransitivityCheck_format_transitivity_cycles_transitivity_break():
    dm = skc.datasets.load_wang2005()
    orank = ws_pipe.evaluate(dm)
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    graph = trans_checker._dominance_graph(dm, orank)
    trans_break = list(nx.simple_cycles(graph, length_bound=3))
    result = skcriteria.ranksrev.transitivity_check._format_transitivity_cycles(trans_break)
    assert result != []

# =============================================================================
# PROPERTIES
# =============================================================================
def test_TransitivityChecker_bad_pipe():
    bad_pipe = "Suffering and pain"
    with pytest.raises(TypeError) as ex:
        skcriteria.ranksrev.transitivity_check.TransitivityChecker(bad_pipe)
    assert "'dmaker' must implement 'evaluate()' method" in str(ex.value)

def test_TransitivityChecker_repr():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    assert repr(trans_checker) == f"<{trans_checker.get_method_name()} {repr(trans_checker.dmaker)}>"

def test_TransitivityChecker_dmaker():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    assert trans_checker.dmaker == ws_pipe

def test_TransitivityChecker_parallell_backend_none():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    assert trans_checker.parallel_backend is None

def test_TransitivityChecker_parallell_backend():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe, parallel_backend=ws_pipe)
    assert trans_checker.parallel_backend == ws_pipe

def test_TransitivityChecker_random_state():
    rnd_state = 42
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe,random_state=rnd_state)
    assert trans_checker.random_state.random() == np.random.default_rng(rnd_state).random()

def test_TransitivityChecker_make_transitivity_strategy_random():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    assert trans_checker.make_transitive_strategy == "random"

def test_TransitivityChecker_make_transitivity_strategy_divination():
    strat = "divination"
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe, make_transitive_strategy=strat)
    assert trans_checker.make_transitive_strategy == strat

def test_TransitivityChecker_max_ranks_default():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    assert trans_checker.max_ranks == 50

def test_TransitivityChecker_max_ranks_custom():
    ranks = 42
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe, max_ranks=ranks)
    assert trans_checker.max_ranks == ranks

def test_TransitivityChecker_n_jobs_default():
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    assert trans_checker.n_jobs is None

def test_TransitivityChecker_n_jobs_custom():
    jobs = 42
    trans_checker = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe, n_jobs=jobs)
    assert trans_checker.n_jobs == jobs

# =============================================================================
# TEST CRITERIA
# =============================================================================

def test_TransitivityCheck_test_criterion_2_pass():
    dm = skc.datasets.load_non_rank_reversal_matrix()
    trans_check = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    rank_comparator = trans_check.evaluate(dm=dm)
    assert rank_comparator._extra.transitivity_break_rate == 0
    assert rank_comparator._extra.test_criterion_2 == True
    assert trans_check._test_criterion_2(rank_comparator._extra.transitivity_break_rate) == True

def test_TransitivityCheck_test_criterion_2_fail():
    dm = skc.datasets.load_wang2005()
    trans_check = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    rank_comparator = trans_check.evaluate(dm=dm)
    assert rank_comparator._extra.transitivity_break_rate > 0
    assert rank_comparator._extra.test_criterion_2 == False
    assert trans_check._test_criterion_2(rank_comparator._extra.transitivity_break_rate) == False

def test_TransitivityCheck_test_criterion_3_pass(): #TODO: no se si esto debería ser asi
    dm = skc.datasets.load_non_rank_reversal_matrix()
    trans_check = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    rank_comparator = trans_check.evaluate(dm=dm)
    assert rank_comparator._extra.test_criterion_3 == True
    assert len(rank_comparator.ranks) == 0

def test_TransitivityCheck_test_criterion_3_fail():
    dm = skc.datasets.load_wang2005()
    trans_check = skcriteria.ranksrev.transitivity_check.TransitivityChecker(ws_pipe)
    rank_comparator = trans_check.evaluate(dm=dm)
    assert rank_comparator._extra.test_criterion_3 == False
    assert len(rank_comparator.ranks) > 0
