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
from skcriteria.agg.similarity import TOPSIS
from skcriteria.ranksrev.rank_invariant_check import RankInvariantChecker
import skcriteria.ranksrev.transitivity_check
from skcriteria.utils import rank

# =============================================================================
# TESTS
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

def test_TransitivityCheck_untie_both():
    first = np.array([1,1,1,1,1])
    second = np.array([0,0,0,0,0])
    actual = skcriteria.ranksrev.transitivity_check._untie_equivalent_ranks(first,second)
    assert actual == ((first,second), (second, first))

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
