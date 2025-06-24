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
    actual = skcriteria.ranksrev.transitivity_check._transitivity_break_bound(
        value
    )
    assert actual == expected


def test_TransitivityCheck_transitivity_break_bound_odd():
    value = 11
    expected = 55
    actual = skcriteria.ranksrev.transitivity_check._transitivity_break_bound(
        value
    )
    assert actual == expected
