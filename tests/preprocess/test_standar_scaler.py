#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocess.standar_scaler.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.preprocess import StandarScaler, scale_by_stdscore


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_StandarScaler_simple_matrix():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [(1 - 2.5) / 1.5, (2 - 3.5) / 1.5, (3 - 4.5) / 1.5],
            [(4 - 2.5) / 1.5, (5 - 3.5) / 1.5, (6 - 4.5) / 1.5],
        ],
        objectives=[min, max, min],
        weights=[1, 2, 3],
        dtypes=[float, float, float],
    )

    scaler = StandarScaler(target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_StandarScaler_matrix(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=(dm.matrix - np.mean(dm.matrix, axis=0, keepdims=True))
        / np.std(dm.matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = StandarScaler(target="matrix")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_StandarScaler_simple_weights():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[
            (1 - 2) / 0.816496580927726,
            (2 - 2) / 0.816496580927726,
            (3 - 2) / 0.816496580927726,
        ],
        dtypes=[int, int, int],
    )

    scaler = StandarScaler(target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_StandarScaler_weights(decision_matrix):

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
        weights=(dm.weights - np.mean(dm.weights)) / np.std(dm.weights),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = StandarScaler(target="weights")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_StandarScaler_simple_both():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [(1 - 2.5) / 1.5, (2 - 3.5) / 1.5, (3 - 4.5) / 1.5],
            [(4 - 2.5) / 1.5, (5 - 3.5) / 1.5, (6 - 4.5) / 1.5],
        ],
        objectives=[min, max, min],
        weights=[
            (1 - 2) / 0.816496580927726,
            (2 - 2) / 0.816496580927726,
            (3 - 2) / 0.816496580927726,
        ],
        dtypes=[float, float, float],
    )

    scaler = StandarScaler(target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_StandarScaler_both(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=(dm.matrix - np.mean(dm.matrix, axis=0, keepdims=True))
        / np.std(dm.matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=(dm.weights - np.mean(dm.weights)) / np.std(dm.weights),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    scaler = StandarScaler(target="both")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_StandarScaler_no_change_original_dm(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    scaler = StandarScaler(target="both")
    dmt = scaler.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_scale_by_stdscore_weights(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nweights = scale_by_stdscore(dm.weights, axis=0)
    expected = (dm.weights - np.mean(dm.weights)) / np.std(dm.weights)

    assert np.all(nweights == expected)


def test_scale_by_stdscore_matrix(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nmtx = scale_by_stdscore(dm.matrix, axis=0)
    expected = (
        dm.matrix - np.mean(dm.matrix, axis=0, keepdims=True)
    ) / np.std(dm.matrix, axis=0, keepdims=True)

    assert np.all(nmtx == expected)
