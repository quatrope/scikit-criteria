#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.invert_objectives"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.preprocessing.invert_objectives import (
    BenefitCostInverter,
    InvertMinimize,
    MinMaxInverter,
    NegateMinimize,
    SKCObjectivesInverterABC,
)

# =============================================================================
# TEST CLASSES ABC
# =============================================================================


def test_SKCObjectivesInverterABC__invert_not_implemented(decision_matrix):
    class Foo(SKCObjectivesInverterABC):
        _skcriteria_parameters = []

        def _invert(self, matrix, minimize_mask):
            return super()._invert(matrix, minimize_mask)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


# =============================================================================
# INVERT
# =============================================================================


def test_NegateMinimize_all_min(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )
    expected = skcriteria.mkdm(
        matrix=-dm.matrix,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
    )

    inv = NegateMinimize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_NegateMinimize_50percent_min(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    minimize_mask = dm.iobjectives == -1
    expected_mtx = np.array(dm.matrix, dtype=float)
    expected_mtx[:, minimize_mask] = -expected_mtx[:, minimize_mask]

    inv_dtypes = np.where(dm.iobjectives == -1, float, dm.dtypes)

    expected = skcriteria.mkdm(
        matrix=expected_mtx,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=inv_dtypes,
    )

    inv = NegateMinimize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_NegateMinimize_no_change_original_dm(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    inv = NegateMinimize()
    dmt = inv.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


# =============================================================================
# INVERT
# =============================================================================


def test_InvertMinimize_all_min(decision_matrix):
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
        alternatives=dm.alternatives,
        criteria=dm.criteria,
    )

    inv = InvertMinimize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_InvertMinimize_50percent_min(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    minimize_mask = dm.iobjectives == -1
    expected_mtx = np.array(dm.matrix, dtype=float)
    expected_mtx[:, minimize_mask] = 1.0 / expected_mtx[:, minimize_mask]

    inv_dtypes = np.where(dm.iobjectives == -1, float, dm.dtypes)

    expected = skcriteria.mkdm(
        matrix=expected_mtx,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=inv_dtypes,
    )

    inv = InvertMinimize()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_InvertMinimize_no_change_original_dm(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    inv = InvertMinimize()
    dmt = inv.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


# =============================================================================
# MIN-MAX INVERT
# =============================================================================


def test_MinMaxInverter_all_min(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    matrix = np.array(dm.matrix, dtype=float)
    maxs = np.max(matrix, axis=0)
    mins = np.min(matrix, axis=0)

    expected_mtx = (matrix - maxs) / (mins - maxs)

    expected = skcriteria.mkdm(
        matrix=expected_mtx,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=np.full(20, float),
    )

    inv = MinMaxInverter()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_MinMaxInverter_50percent_min(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    minimize_mask = dm.iobjectives == -1
    benefit_mask = ~minimize_mask

    matrix = np.array(dm.matrix, dtype=float)
    expected_mtx = np.empty_like(matrix)

    maxs = np.max(matrix, axis=0)
    mins = np.min(matrix, axis=0)

    expected_mtx[:, minimize_mask] = (
        matrix[:, minimize_mask] - maxs[minimize_mask]
    ) / (mins[minimize_mask] - maxs[minimize_mask])

    expected_mtx[:, benefit_mask] = (
        matrix[:, benefit_mask] - mins[benefit_mask]
    ) / (maxs[benefit_mask] - mins[benefit_mask])

    expected = skcriteria.mkdm(
        matrix=expected_mtx,
        objectives=np.full(20, 1, dtype=int),
        weights=dm.weights,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=np.full(20, float),
    )

    inv = MinMaxInverter()

    result = inv.transform(dm)

    assert result.equals(expected)


def test_MinMaxInverter_no_change_original_dm(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = dm.copy()

    inv = MinMaxInverter()
    dmt = inv.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


@pytest.mark.parametrize("objective", [min, max])
def test_MinMaxInverter_constant_criterion(objective):
    dm = skcriteria.mkdm(
        matrix=[[1, 2], [1, 4], [1, 6]],
        objectives=[objective, max],
    )

    inv = MinMaxInverter()
    with pytest.warns(UserWarning):
        dmt = inv.transform(dm)

    dmdict = dmt.to_dict()
    mtx = dmdict["matrix"]

    assert (mtx[:, 0] == 0).all()
    assert np.all(dmdict["objectives"] == 1)


# =============================================================================
# BENEFIT-COST INVERT
# =============================================================================


def test_BenefitCostInverter_negative_value():
    dm = skcriteria.mkdm(
        matrix=[[-1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
    )

    inv = BenefitCostInverter()

    with pytest.raises(ValueError):
        inv.transform(dm)


def test_BenefitCostInverter_max_value_zero():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [3, 0, 6]],
        objectives=[min, max, min],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 0, 1], [0.333, 0, 0.5]],
        objectives=[min, max, min],
    )

    inv = BenefitCostInverter()

    dm_transformed = inv.transform(dm)

    assert np.allclose(
        dm_transformed.matrix.to_numpy(), expected.matrix.to_numpy(), atol=1e-3
    )


def test_BenefitCostInverter_min_value_zero():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
    )

    expected = skcriteria.mkdm(
        matrix=[[0, 0, 1], [0, 1, 0.5]],
        objectives=[min, max, min],
    )

    inv = BenefitCostInverter()

    dm_transformed = inv.transform(dm)

    assert np.allclose(
        dm_transformed.matrix.to_numpy(), expected.matrix.to_numpy(), atol=1e-3
    )


def test_BenefitCostInverter_lisco():
    """
    Data From:
        Badi, I., Shetwan, A. G., & Abdulshahed, A. M. (2017, September).
        Supplier selection using COmbinative Distance-based
        ASsessment (CODAS) method for multi-criteria decision-making.
        In Proceedings of the 1st international conference on management,
        engineering and environment (ICMNEE) (pp. 395-407).
    """
    dm = skcriteria.mkdm(
        matrix=[
            [45, 3600, 45, 0.9],
            [25, 3800, 60, 0.8],
            [23, 3100, 35, 0.9],
            [14, 3400, 50, 0.7],
            [15, 3300, 40, 0.8],
            [28, 3000, 30, 0.6],
        ],
        objectives=[max, min, max, max],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [1.000, 0.833, 0.750, 1.000],
            [0.556, 0.789, 1.000, 0.889],
            [0.511, 0.968, 0.583, 1.000],
            [0.311, 0.882, 0.833, 0.778],
            [0.333, 0.909, 0.667, 0.889],
            [0.622, 1.000, 0.500, 0.667],
        ],
        objectives=[max, min, max, max],
    )

    inv = BenefitCostInverter()

    dm_transformed = inv.transform(dm)

    assert np.allclose(
        dm_transformed.matrix.to_numpy(), expected.matrix.to_numpy(), atol=1e-3
    )
