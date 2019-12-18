#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.data

"""


# =============================================================================
# IMPORTS
# =============================================================================

import string

import numpy as np

import pytest

from skcriteria import Data, ascriteria


# =============================================================================
# CONSTANTS
# =============================================================================

CHARS = tuple(string.ascii_letters + string.digits + string.punctuation)

RANDOM = np.random.RandomState(42)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def init_data():
    random = RANDOM

    alts = random.randint(2, 10)
    crit = random.randint(2, 10)

    mtx = random.rand(alts, crit)

    criteria = random.choice([min, max, 1, -1, np.min, np.max], crit)
    weights = random.rand(crit)

    anames = ["".join(random.choice(CHARS, 15)) for _ in range(alts)]
    cnames = ["".join(random.choice(CHARS, 15)) for _ in range(crit)]

    return mtx, criteria, weights, anames, cnames


# =============================================================================
# TESTS
# =============================================================================

def test_simple_creation(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)

    assert np.all(data.mtx == mtx)
    assert np.all(data.criteria == ascriteria(criteria))
    assert np.all(data.weights == weights)
    assert data.anames == tuple(anames)
    assert data.cnames == tuple(cnames)
