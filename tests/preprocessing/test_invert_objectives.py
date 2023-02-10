#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
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

import pytest

import skcriteria
from skcriteria.preprocessing.invert_objectives import (
    InvertMinimize,
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
