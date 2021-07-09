#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocess.sum_scaler.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.preprocess import SumScaler, scale_by_sum


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_SumScaler_simple_matrix():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1 / 5, 2 / 7, 3 / 9], [4 / 5, 5 / 7, 6 / 9]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
        dtypes=[float, float, float],
    )

    scaler = SumScaler(target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_SumScaler_matrix(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix
        / np.sum(dm.matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = SumScaler(target="matrix")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_SumScaler_simple_weights():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1 / 6, 2 / 6, 3 / 6],
        dtypes=[int, int, int],
    )

    scaler = SumScaler(target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_SumScaler_weights(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=dm.weights / np.sum(dm.weights),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = SumScaler(target="weights")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_SumScaler_simple_both():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1 / 5, 2 / 7, 3 / 9], [4 / 5, 5 / 7, 6 / 9]],
        objectives=[min, max, min],
        weights=[1 / 6, 2 / 6, 3 / 6],
        dtypes=[float, float, float],
    )

    scaler = SumScaler(target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_SumScaler_both(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix
        / np.sum(dm.matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights / np.sum(dm.weights),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = SumScaler(target="both")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_SumScaler_no_change_original_dm(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    scaler = SumScaler(target="both")
    dmt = scaler.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_scale_by_sum_weights(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nweights = scale_by_sum(dm.weights, axis=0)
    expected = dm.weights / np.sum(dm.weights)

    assert np.all(nweights == expected)


def test_scale_by_sum_matrix(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nmtx = scale_by_sum(dm.matrix, axis=0)
    expected = dm.matrix / np.sum(
        dm.matrix, axis=0, keepdims=True, dtype=float
    )

    assert np.all(nmtx == expected)
