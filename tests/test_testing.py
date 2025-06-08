#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

"""Tests for skcriteria/testing.py"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria import agg, cmp, testing


# =============================================================================
# assert_dmatrix_equals
# =============================================================================


# test skcriteria.testing.assert_dmatrix_equals
def test_assert_dmatrix_equals(decision_matrix):
    left = decision_matrix(seed=42)
    right = left.copy()
    testing.assert_dmatrix_equals(left, right)


def test_assert_dmatrix_equals_same_object(decision_matrix):
    dm = decision_matrix(seed=42)
    testing.assert_dmatrix_equals(dm, dm)


def test_assert_dmatrix_equals_not_dmatrix(decision_matrix):
    dm = decision_matrix(seed=42)
    with pytest.raises(
        AssertionError,
        match=(
            "'left' is not a DecisionMatrix instance. "
            "Found <class 'NoneType'>"
        ),
    ):
        testing.assert_dmatrix_equals(None, dm)
    with pytest.raises(
        AssertionError,
        match=(
            "'right' is not a DecisionMatrix instance. "
            "Found <class 'NoneType'>"
        ),
    ):
        testing.assert_dmatrix_equals(dm, None)


def test_assert_dmatrix_equals_not_same_alternatives(decision_matrix):
    left = decision_matrix(seed=42)

    alternatives = list(left.alternatives)
    alternatives[0] = alternatives[0] + "_foo"

    right = left.replace(alternatives=alternatives)

    with pytest.raises(AssertionError):
        testing.assert_dmatrix_equals(left, right)


def test_assert_dmatrix_equals_not_same_criteria(decision_matrix):
    left = decision_matrix(seed=42)

    criteria = list(left.criteria)
    criteria[0] = criteria[0] + "_foo"

    right = left.replace(criteria=criteria)

    with pytest.raises(AssertionError):
        testing.assert_dmatrix_equals(left, right)


def test_assert_dmatrix_equals_not_same_matrix(decision_matrix):
    left = decision_matrix(seed=42)

    matrix = left.matrix + 1

    right = left.replace(matrix=matrix)

    with pytest.raises(AssertionError):
        testing.assert_dmatrix_equals(left, right)


def test_assert_dmatrix_equals_not_same_objectives(decision_matrix):
    left = decision_matrix(seed=42)

    objectives = list(left.iobjectives)
    objectives[0] = objectives[0] * -1

    right = left.replace(objectives=objectives)

    with pytest.raises(AssertionError):
        testing.assert_dmatrix_equals(left, right)


def test_assert_dmatrix_equals_not_same_weights(decision_matrix):
    left = decision_matrix(seed=42)

    weights = list(left.weights)
    weights[0] = weights[0] + 1

    right = left.replace(weights=weights)

    with pytest.raises(AssertionError):
        testing.assert_dmatrix_equals(left, right)


def test_assert_dmatrix_equals_not_same_dtypes(decision_matrix):
    left = decision_matrix(seed=42)
    right = left.replace(dtypes=[np.float32] * len(left.criteria))

    with pytest.raises(AssertionError):
        testing.assert_dmatrix_equals(left, right, check_dtypes=True)


# =============================================================================
# assert_result_equals
# =============================================================================


def test_assert_result_equals():
    rresult_left = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rresult_right = agg.RankResult("test", ["a", "b"], [1, 1], {})

    testing.assert_result_equals(rresult_left, rresult_right)

    kresult_left = agg.KernelResult("test", ["a", "b"], [True, False], {})
    kresult_right = agg.KernelResult("test", ["a", "b"], [True, False], {})

    testing.assert_result_equals(kresult_left, kresult_right)


def test_assert_result_equals_same_object():
    rresult = agg.RankResult("test", ["a", "b"], [1, 1], {})
    testing.assert_result_equals(rresult, rresult)

    kresult = agg.KernelResult("test", ["a", "b"], [True, False], {})
    testing.assert_result_equals(kresult, kresult)


def test_assert_result_equals_not_result():
    rresult = agg.RankResult("test", ["a", "b"], [1, 1], {})
    with pytest.raises(
        AssertionError,
        match="'right' is not a ResultABC instance. Found <class 'NoneType'>",
    ):
        testing.assert_result_equals(rresult, None)
    with pytest.raises(
        AssertionError,
        match="'left' is not a ResultABC instance. Found <class 'NoneType'>",
    ):
        testing.assert_result_equals(None, rresult)

    kresult = agg.KernelResult("test", ["a", "b"], [True, False], {})
    with pytest.raises(
        AssertionError,
        match="'right' is not a ResultABC instance. Found <class 'NoneType'>",
    ):
        testing.assert_result_equals(kresult, None)
    with pytest.raises(
        AssertionError,
        match="'left' is not a ResultABC instance. Found <class 'NoneType'>",
    ):
        testing.assert_result_equals(None, kresult)


def test_assert_result_equals_not_same_type_of_result():
    rresult = agg.RankResult("test", ["a", "b"], [1, 1], {})
    kresult = agg.KernelResult("test", ["a", "b"], [True, False], {})
    with pytest.raises(AssertionError):
        testing.assert_result_equals(rresult, kresult)


def test_assert_result_equals_not_same_alternatives():
    rresult_left = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rresult_right = agg.RankResult("test", ["a", "c"], [1, 1], {})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(rresult_left, rresult_right)

    kresult_left = agg.KernelResult("test", ["a", "b"], [True, False], {})
    kresult_right = agg.KernelResult("test", ["a", "c"], [True, False], {})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(kresult_left, kresult_right)


def test_assert_result_equals_not_same_method():
    rresult_left = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rresult_right = agg.RankResult("test_b", ["a", "b"], [1, 1], {})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(rresult_left, rresult_right)

    kresult_left = agg.KernelResult("test", ["a", "b"], [True, False], {})
    kresult_right = agg.KernelResult("test_b", ["a", "b"], [True, False], {})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(kresult_left, kresult_right)


def test_assert_result_equals_not_same_values():
    rresult_left = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rresult_right = agg.RankResult("test", ["a", "b"], [1, 2], {})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(rresult_left, rresult_right)

    kresult_left = agg.KernelResult("test", ["a", "b"], [True, False], {})
    kresult_right = agg.KernelResult("test", ["a", "b"], [True, True], {})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(kresult_left, kresult_right)


def test_assert_result_equals_not_same_extra():
    rresult_left = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rresult_right = agg.RankResult("test", ["a", "b"], [1, 1], {"foo": 1})

    with pytest.raises(AssertionError):
        testing.assert_result_equals(rresult_left, rresult_right)

    kresult_left = agg.KernelResult("test", ["a", "b"], [True, False], {})
    kresult_right = agg.KernelResult(
        "test", ["a", "b"], [True, False], {"foo": 1}
    )

    with pytest.raises(AssertionError):
        testing.assert_result_equals(kresult_left, kresult_right)


# =============================================================================
# assert_rcmp_equals
# =============================================================================


def test_assert_rcmp_equals():
    left = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    right = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )

    testing.assert_rcmp_equals(left, right)


def test_assert_rcmp_equals_same_object():
    rcmp = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    testing.assert_rcmp_equals(rcmp, rcmp)


def test_assert_rcmp_equals_not_RankComparator():
    rcmp = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    err_msg = (
        "'right' is not a RanksComparator instance. Found <class 'NoneType'>"
    )
    with pytest.raises(
        AssertionError,
        match=err_msg,
    ):
        testing.assert_rcmp_equals(rcmp, None)

    err_msg = (
        "'left' is not a RanksComparator instance. Found <class 'NoneType'>"
    )
    with pytest.raises(
        AssertionError,
        match=err_msg,
    ):
        testing.assert_rcmp_equals(None, rcmp)


def test_assert_rcmp_equals_not_same_length():
    left = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    right = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    msg = "RanksComparator instances have different lengths: 2 != 3"
    with pytest.raises(AssertionError, match=msg):
        testing.assert_rcmp_equals(left, right)


def test_assert_rcmp_equals_not_same_ranks():
    left = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    right = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 2], {}),
    )
    with pytest.raises(AssertionError):
        testing.assert_rcmp_equals(left, right)
