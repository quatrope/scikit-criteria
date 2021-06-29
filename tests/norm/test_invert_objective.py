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


def test_MinimizeToMaximize_simple(decision_matrix):

    dm = skcriteria.mkdm(
        mtx=[[1, 2, 3], [4, 5, 6]], objectives=[min, max, min]
    )

    expected = skcriteria.mkdm(
        mtx=[[1, 2, 1 / 3], [1 / 4, 5, 1 / 6]], objectives=[max, max, max]
    )

    normalizer = invert_objective.MinimizeToMaximize()

    result = normalizer.normalize(dm)

    assert result == expected


def test_MinimizeToMaximize_all_min(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )
    normalizer = invert_objective.MinimizeToMaximize()

    result = normalizer.normalize(dm)

    assert np.all(dm.objectives_values == -1)
    assert np.all(result.objectives_values == 1)
    assert np.all(result.mtx == 1.0 / dm.mtx)


def test_MinimizeToMaximize_50percent_min(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    normalizer = invert_objective.MinimizeToMaximize()

    result = normalizer.normalize(dm)

    original_maximize = dm.mtx[:, dm.objectives_values == 1]
    original_minimize = dm.mtx[:, dm.objectives_values == -1]

    result_maximize = result.mtx[:, dm.objectives_values == 1]
    result_minimize = result.mtx[:, dm.objectives_values == -1]

    assert np.all(result_maximize == original_maximize)
    assert np.all(result_minimize == 1 / original_minimize)
    assert np.all(result.objectives_values == 1)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_minimize_to_maximize_all_min(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    rmtx, robjs = invert_objective.minimize_to_maximize(
        dm.mtx, dm.objectives_values
    )

    assert np.all(dm.objectives_values == -1)
    assert np.all(robjs == 1)
    assert np.all(rmtx == 1.0 / dm.mtx)


def test_minimize_to_maximize_50percent_min(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    rmtx, robjs = invert_objective.minimize_to_maximize(
        dm.mtx, dm.objectives_values
    )

    original_maximize = dm.mtx[:, dm.objectives_values == 1]
    original_minimize = dm.mtx[:, dm.objectives_values == -1]

    result_maximize = rmtx[:, dm.objectives_values == 1]
    result_minimize = rmtx[:, dm.objectives_values == -1]

    assert np.all(result_maximize == original_maximize)
    assert np.all(result_minimize == 1 / original_minimize)
    assert np.all(robjs == 1)
