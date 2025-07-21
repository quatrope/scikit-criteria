#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.minmax_scaler"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
import skcriteria.testing as skct
from skcriteria.preprocessing.scalers import (
    CenitDistanceMatrixScaler,
    MaxAbsScaler,
    MinMaxScaler,
    StandarScaler,
    SumScaler,
    VectorScaler,
)


# =============================================================================
# TEST MIN MAX
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

    skct.assert_dmatrix_equals(result, expected)


def test_MinMaxScaler_matrix(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    mtx = dm.matrix.to_numpy()
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)

    expected = skcriteria.mkdm(
        matrix=(mtx - mtx_min) / (mtx_max - mtx_min),
        objectives=dm.objectives,
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="matrix")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(result, expected)


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
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="weights")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(result, expected)


def test_MinMaxScaler_both(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    mtx = dm.matrix.to_numpy()
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)

    expected = skcriteria.mkdm(
        matrix=(mtx - mtx_min) / (mtx_max - mtx_min),
        objectives=dm.objectives,
        weights=(dm.weights - np.min(dm.weights))
        / (np.max(dm.weights) - np.min(dm.weights)),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="both")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MinMaxScaler_no_change_original_dm(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    scaler = MinMaxScaler(target="both")
    dmt = scaler.transform(dm)

    skct.assert_dmatrix_equals(dm, expected)
    assert not dmt.equals(expected) and dm is not expected


# =============================================================================
# TEST Standar scaler
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

    skct.assert_dmatrix_equals(result, expected)


def test_StandarScaler_matrix(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=(matrix - np.mean(matrix, axis=0, keepdims=True))
        / np.std(matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = StandarScaler(target="matrix")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(result, expected)


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
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = StandarScaler(target="weights")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(result, expected)


def test_StandarScaler_both(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=(matrix - np.mean(matrix, axis=0, keepdims=True))
        / np.std(matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=(dm.weights - np.mean(dm.weights)) / np.std(dm.weights),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = StandarScaler(target="both")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(dm, expected)
    assert not dmt.equals(expected) and dm is not expected


# =============================================================================
# TEST VECTOR SCALER
# =============================================================================


def test_VectorScaler_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [1 / 4.123105626, 2 / 5.385164807, 3 / 6.708203932],
            [4 / 4.123105626, 5 / 5.385164807, 6 / 6.708203932],
        ],
        objectives=[min, max, min],
        weights=[1, 2, 3],
        dtypes=[float, float, float],
    )

    scaler = VectorScaler(target="matrix")

    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_VectorScaler_matrix(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix / np.sqrt(np.sum(np.power(dm.matrix, 2), axis=0)),
        objectives=dm.objectives,
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = VectorScaler(target="matrix")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_VectorScaler_simple_weights():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1 / 3.741657387, 2 / 3.741657387, 3 / 3.741657387],
        dtypes=[int, int, int],
    )

    scaler = VectorScaler(target="weights")

    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_VectorScaler_weights(decision_matrix):
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
        weights=dm.weights / np.sqrt(np.sum(np.power(dm.weights, 2))),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = VectorScaler(target="weights")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_VectorScaler_simple_both():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [1 / 4.123105626, 2 / 5.385164807, 3 / 6.708203932],
            [4 / 4.123105626, 5 / 5.385164807, 6 / 6.708203932],
        ],
        objectives=[min, max, min],
        weights=[1 / 3.741657387, 2 / 3.741657387, 3 / 3.741657387],
        dtypes=[float, float, float],
    )

    scaler = VectorScaler(target="both")

    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_VectorScaler_both(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix / np.sqrt(np.sum(np.power(dm.matrix, 2), axis=0)),
        objectives=dm.objectives,
        weights=dm.weights / np.sqrt(np.sum(np.power(dm.weights, 2))),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = VectorScaler(target="both")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_VectorScaler_no_change_original_dm(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    scaler = VectorScaler(target="both")
    dmt = scaler.transform(dm)

    skct.assert_dmatrix_equals(dm, expected)
    assert not dmt.equals(expected) and dm is not expected


# =============================================================================
# TEST SUM SCALER
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

    skct.assert_dmatrix_equals(result, expected)


def test_SumScaler_matrix(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=matrix / np.sum(matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = SumScaler(target="matrix")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(result, expected)


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
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = SumScaler(target="weights")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(result, expected)


def test_SumScaler_both(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=matrix / np.sum(matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights / np.sum(dm.weights),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = SumScaler(target="both")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


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

    skct.assert_dmatrix_equals(dm, expected)
    assert not dmt.equals(expected) and dm is not expected


# =============================================================================
# TEST MAX_SCALER
# =============================================================================


def test_MaxAbsScaler_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1 / 4, 2 / 5, 3 / 6], [4 / 4, 5 / 5, 6 / 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
        dtypes=[float, float, float],
    )

    scaler = MaxAbsScaler(target="matrix")

    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MaxAbsScaler_matrix(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=matrix / np.max(matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MaxAbsScaler(target="matrix")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MaxAbsScaler_simple_weights():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1 / 3, 2 / 3, 3 / 3],
        dtypes=[int, int, int],
    )

    scaler = MaxAbsScaler(target="weights")

    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MaxAbsScaler_weights(decision_matrix):
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
        weights=dm.weights / np.max(dm.weights),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MaxAbsScaler(target="weights")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MaxAbsScaler_simple_both():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1 / 4, 2 / 5, 3 / 6], [4 / 4, 5 / 5, 6 / 6]],
        objectives=[min, max, min],
        weights=[1 / 3, 2 / 3, 3 / 3],
        dtypes=[float, float, float],
    )

    scaler = MaxAbsScaler(target="both")

    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MaxAbsScaler_both(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=matrix / np.max(matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=dm.weights / np.max(dm.weights),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MaxAbsScaler(target="both")
    result = scaler.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_MaxAbsScaler_no_change_original_dm(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    scaler = MaxAbsScaler(target="both")
    dmt = scaler.transform(dm)

    skct.assert_dmatrix_equals(dm, expected)
    assert not dmt.equals(expected) and dm is not expected


def test_CenitDistanceMatrixScaler_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skcriteria.mkdm(
        matrix=[[-0.0, 0.0, 1.0], [1.0, 1.0, -0.0]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    tfm = CenitDistanceMatrixScaler()

    result = tfm.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_CenitDistanceMatrixScaler_diakoulaki1995determining():
    """
    Data from:
        Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        Determining objective weights in multiple criteria problems:
        The critic method. Computers & Operations Research, 22(7), 763-770.
    """

    dm = skcriteria.mkdm(
        matrix=[
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98],
        ],
        objectives=[max, max, max],
        weights=[61, 1.08, 4.33],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [1.0, 0.21243523, 0.99702381],
            [0.57845188, 0.0, 1.0],
            [0.53242678, 0.44559585, 0.46130952],
            [0.45606695, 0.78497409, 0.19940476],
            [0.41841004, 0.65025907, 0.40178571],
            [0.40376569, 1.0, 0.06845238],
            [0.29811715, 0.84455959, 0.33333333],
            [0.0, 0.79015544, 0.0],
        ],
        objectives=[max, max, max],
        weights=[61, 1.08, 4.33],
    )

    tfm = CenitDistanceMatrixScaler()
    result = tfm.transform(dm)

    skct.assert_dmatrix_equals(result, expected)


def test_CenitDistanceMatrixScaler_no_change_original_dm():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = dm.copy()

    tfm = CenitDistanceMatrixScaler()
    dmt = tfm.transform(dm)

    skct.assert_dmatrix_equals(dm, expected)
    assert not dmt.equals(expected) and dm is not expected
