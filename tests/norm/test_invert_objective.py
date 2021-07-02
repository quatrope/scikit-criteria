#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.norm.invert_objective

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


import skcriteria
from skcriteria.norm import invert_objective


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_MinimizeToMaximizeNormalizer_simple():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]], objectives=[min, max, min]
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 1 / 3], [1 / 4, 5, 1 / 6]], objectives=[max, max, max]
    )

    normalizer = invert_objective.MinimizeToMaximizeNormalizer()

    result = normalizer.normalize(dm)

    assert result.equals(expected)


def test_MinimizeToMaximizeNormalizer_all_min(decision_matrix):

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

    normalizer = invert_objective.MinimizeToMaximizeNormalizer()

    result = normalizer.normalize(dm)

    assert result.equals(expected)


def test_MinimizeToMaximizeNormalizer_50percent_min(decision_matrix):

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

    normalizer = invert_objective.MinimizeToMaximizeNormalizer()

    result = normalizer.normalize(dm)

    assert result.equals(expected)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_invert_all_min(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    rmtx = invert_objective.invert(
        dm.matrix, dm.objectives_values == skcriteria.Objective.MIN.value
    )

    assert np.all(dm.objectives_values == -1)
    assert np.all(rmtx == 1.0 / dm.matrix)


def test_invert_50percent_min(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    rmtx = invert_objective.invert(
        dm.matrix, dm.objectives_values == skcriteria.Objective.MIN.value
    )

    original_maximize = dm.matrix[:, dm.objectives_values == 1]
    original_minimize = dm.matrix[:, dm.objectives_values == -1]

    result_maximize = rmtx[:, dm.objectives_values == 1]
    result_minimize = rmtx[:, dm.objectives_values == -1]

    assert np.all(result_maximize == original_maximize)
    assert np.all(result_minimize == 1 / original_minimize)
