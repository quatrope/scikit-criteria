#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocess.minmax_scaler

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.preprocess import MinMaxScaler, scale_by_minmax


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_MinMaxScaler_simple_matrix():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [(1 - 1) / (4 - 1), (2 - 2) / (5 - 2), (3 - 3) / (6 - 3)],
            [(4 - 1) / (4 - 1), (5 - 2) / (5 - 2), (6 - 3) / (6 - 3)],
        ],
        objectives=[min, max, min],
        weights=[1, 2, 3],
        dtypes=[float, float, float],
    )

    scaler = MinMaxScaler(target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MinMaxScaler_matrix(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    mtx = dm.matrix
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)

    expected = skcriteria.mkdm(
        matrix=(mtx - mtx_min) / (mtx_max - mtx_min),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="matrix")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MinMaxScaler_simple_weights():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[(1 - 1) / (3 - 1), (2 - 1) / (3 - 1), (3 - 1) / (3 - 1)],
        dtypes=[int, int, int],
    )

    scaler = MinMaxScaler(target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MinMaxScaler_weights(decision_matrix):

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
        weights=(dm.weights - np.min(dm.weights))
        / (np.max(dm.weights) - np.min(dm.weights)),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="weights")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MinMaxScaler_simple_both():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [(1 - 1) / (4 - 1), (2 - 2) / (5 - 2), (3 - 3) / (6 - 3)],
            [(4 - 1) / (4 - 1), (5 - 2) / (5 - 2), (6 - 3) / (6 - 3)],
        ],
        objectives=[min, max, min],
        weights=[(1 - 1) / (3 - 1), (2 - 1) / (3 - 1), (3 - 1) / (3 - 1)],
        dtypes=[float, float, float],
    )

    scaler = MinMaxScaler(target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MinMaxScaler_both(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    mtx = dm.matrix
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)

    expected = skcriteria.mkdm(
        matrix=(mtx - mtx_min) / (mtx_max - mtx_min),
        objectives=dm.objectives,
        weights=(dm.weights - np.min(dm.weights))
        / (np.max(dm.weights) - np.min(dm.weights)),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="both")
    result = scaler.transform(dm)

    assert result.equals(expected)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_scale_by_minmax_weights(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nweights = scale_by_minmax(dm.weights, axis=0)
    expected = (
        (dm.weights - np.min(dm.weights))
        / (np.max(dm.weights) - np.min(dm.weights)),
    )

    assert np.all(nweights == expected)


def test_scale_by_minmax_matrix(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nmtx = scale_by_minmax(dm.matrix, axis=0)

    mtx = dm.matrix
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)
    expected = ((mtx - mtx_min) / (mtx_max - mtx_min),)

    assert np.all(nmtx == expected)
