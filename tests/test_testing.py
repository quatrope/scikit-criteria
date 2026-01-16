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


def test_assert_result_equals_skip_extra():
    # Test that skip_extra=True allows different extra_ attributes
    rresult_left = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rresult_right = agg.RankResult("test", ["a", "b"], [1, 1], {"foo": 1})

    # Should not raise when skip_extra=True
    testing.assert_result_equals(rresult_left, rresult_right, skip_extra=True)

    kresult_left = agg.KernelResult("test", ["a", "b"], [True, False], {})
    kresult_right = agg.KernelResult(
        "test", ["a", "b"], [True, False], {"foo": 1}
    )

    # Should not raise when skip_extra=True
    testing.assert_result_equals(kresult_left, kresult_right, skip_extra=True)

    # Test with more complex extra_ differences
    rresult_left_complex = agg.RankResult(
        "test", ["a", "b"], [1, 2], {"meta": "data1", "value": 42}
    )
    rresult_right_complex = agg.RankResult(
        "test", ["a", "b"], [1, 2], {"meta": "data2", "value": 100}
    )

    # Should not raise when skip_extra=True
    testing.assert_result_equals(
        rresult_left_complex, rresult_right_complex, skip_extra=True
    )


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


def test_assert_rcmp_equals_skip_extra():
    # Test that skip_extra=True allows different extra_ attributes in ranks
    # left = cmp.mkrank_cmp(
    #     agg.RankResult("test", ["a", "b"], [1, 1], {"meta": "left"}),
    #     agg.RankResult("test", ["a", "b"], [1, 2], {"info": "foo"}),
    # )
    # right = cmp.mkrank_cmp(
    #     agg.RankResult("test", ["a", "b"], [1, 1], {"meta": "right"}),
    #     agg.RankResult("test", ["a", "b"], [1, 2], {"info": "bar"}),
    # )

    # # Should not raise when skip_extra=True
    # testing.assert_rcmp_equals(left, right, skip_extra=True)

    # # Test with different extra_ in RanksComparator itself
    # left_rcmp_extra = cmp.mkrank_cmp(
    #     agg.RankResult("test", ["a", "b"], [1, 1], {}),
    #     agg.RankResult("test", ["a", "b"], [1, 2], {}),
    #     extra={"rcmp_meta": "left_data"},
    # )
    # right_rcmp_extra = cmp.mkrank_cmp(
    #     agg.RankResult("test", ["a", "b"], [1, 1], {}),
    #     agg.RankResult("test", ["a", "b"], [1, 2], {}),
    #     extra={"rcmp_meta": "right_data"},
    # )

    # # Should not raise when skip_extra=True
    # (skips RanksComparator extra_ too)
    # testing.assert_rcmp_equals(
    #     left_rcmp_extra, right_rcmp_extra, skip_extra=True
    # )

    # Should still fail if core attributes differ
    left_diff = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {"meta": "left"}),
        agg.RankResult("test", ["a", "b"], [1, 2], {"info": "foo"}),
    )
    right_diff = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {"meta": "right"}),
        agg.RankResult("test", ["a", "b"], [2, 1], {"info": "bar"}),
    )

    # Should raise even with skip_extra=True because values differ
    with pytest.raises(AssertionError):
        testing.assert_rcmp_equals(left_diff, right_diff, skip_extra=True)

    left_diff = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 2], {}),
        extra={"meta": "left"},
    )
    right_diff = cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 2], {}),
        extra={"meta": "right"},
    )

    # Should raise even with skip_extra=True because values differ
    with pytest.raises(AssertionError):
        testing.assert_rcmp_equals(left_diff, right_diff, skip_extra=False)
