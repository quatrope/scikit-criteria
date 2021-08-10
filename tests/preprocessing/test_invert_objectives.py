#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.invert_objectives

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


import skcriteria
from skcriteria.preprocessing import MinimizeToMaximize, invert


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_MinimizeToMaximize_simple():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]], objectives=[min, max, min]
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 1 / 3], [1 / 4, 5, 1 / 6]], objectives=[max, max, max]
    )

    inv = MinimizeToMaximize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_MinimizeToMaximize_all_min(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )
    expected = skcriteria.mkdm(
        matrix=1.0 / dm.matrix,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
    )

    inv = MinimizeToMaximize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_MinimizeToMaximize_50percent_min(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    minimize_mask = dm.objectives_values == -1
    expected_mtx = np.array(dm.matrix, dtype=float)
    expected_mtx[:, minimize_mask] = 1.0 / expected_mtx[:, minimize_mask]

    inv_dtypes = np.where(dm.objectives_values == -1, float, dm.dtypes)

    expected = skcriteria.mkdm(
        matrix=expected_mtx,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=inv_dtypes,
    )

    inv = MinimizeToMaximize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_MinimizeToMaximize_no_change_original_dm(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    inv = MinimizeToMaximize()
    dmt = inv.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )
