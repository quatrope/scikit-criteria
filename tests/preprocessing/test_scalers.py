#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.minmax_scaler

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.preprocessing import (
    MaxScaler,
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

    mtx = dm.matrix.to_numpy()
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)

    expected = skcriteria.mkdm(
        matrix=(mtx - mtx_min) / (mtx_max - mtx_min),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        criteria=dm.criteria,
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
        criteria=dm.criteria,
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

    mtx = dm.matrix.to_numpy()
    mtx_min = np.min(mtx, axis=0, keepdims=True)
    mtx_max = np.max(mtx, axis=0, keepdims=True)

    expected = skcriteria.mkdm(
        matrix=(mtx - mtx_min) / (mtx_max - mtx_min),
        objectives=dm.objectives,
        weights=(dm.weights - np.min(dm.weights))
        / (np.max(dm.weights) - np.min(dm.weights)),
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MinMaxScaler(target="both")
    result = scaler.transform(dm)

    assert result.equals(expected)


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

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


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

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=(matrix - np.mean(matrix, axis=0, keepdims=True))
        / np.std(matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        criteria=dm.criteria,
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
        criteria=dm.criteria,
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

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=(matrix - np.mean(matrix, axis=0, keepdims=True))
        / np.std(matrix, axis=0, keepdims=True),
        objectives=dm.objectives,
        weights=(dm.weights - np.mean(dm.weights)) / np.std(dm.weights),
        anames=dm.anames,
        criteria=dm.criteria,
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

    assert result.aequals(expected)


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
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = VectorScaler(target="matrix")
    result = scaler.transform(dm)

    assert result.aequals(expected)


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

    assert result.aequals(expected)


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
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = VectorScaler(target="weights")
    result = scaler.transform(dm)

    assert result.aequals(expected)


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

    assert result.aequals(expected)


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
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = VectorScaler(target="both")
    result = scaler.transform(dm)

    assert result.aequals(expected)


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

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


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

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=matrix / np.sum(matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        criteria=dm.criteria,
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
        criteria=dm.criteria,
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

    matrix = dm.matrix.to_numpy()
    expected = skcriteria.mkdm(
        matrix=matrix / np.sum(matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights / np.sum(dm.weights),
        anames=dm.anames,
        criteria=dm.criteria,
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
# TEST MAX_SCALER
# =============================================================================


def test_MaxScaler_simple_matrix():

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

    scaler = MaxScaler(target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MaxScaler_matrix(decision_matrix):

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
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MaxScaler(target="matrix")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MaxScaler_simple_weights():

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

    scaler = MaxScaler(target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MaxScaler_weights(decision_matrix):

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
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MaxScaler(target="weights")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MaxScaler_simple_both():

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

    scaler = MaxScaler(target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MaxScaler_both(decision_matrix):

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
        anames=dm.anames,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    scaler = MaxScaler(target="both")
    result = scaler.transform(dm)

    assert result.equals(expected)


def test_MaxScaler_no_change_original_dm(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    scaler = MaxScaler(target="both")
    dmt = scaler.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )
