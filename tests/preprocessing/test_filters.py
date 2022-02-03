#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.filters.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pytest

import skcriteria as skc
from skcriteria.preprocessing import filters


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_SKCFilterABC_not_provide_filters():
    class FooFilter(filters.SKCFilterABC):
        def _make_mask(self, matrix, criteria):
            pass

        def _validate_filters(self, filters):
            pass

    with pytest.raises(ValueError):
        FooFilter({})


def test_SKCFilterABC_not_implemented_make_mask():
    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    class FooFilter(filters.SKCFilterABC):
        def _make_mask(self, matrix, criteria):
            return super()._make_mask(matrix, criteria)

        def _validate_filters(self, filters):
            pass

    tfm = FooFilter({"ROE": 1})

    with pytest.raises(NotImplementedError):
        tfm.transform(dm)


def test_SKCFilterABC_not_implemented_validate_filters():
    class FooFilter(filters.SKCFilterABC):
        def _make_mask(self, matrix, criteria):
            pass

        def _validate_filters(self, filters):
            return super()._validate_filters(filters)

    with pytest.raises(NotImplementedError):
        FooFilter({"ROE": 1})


def test_SKCFilterABC_missing_criteria():
    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    class FooFilter(filters.SKCFilterABC):
        def _make_mask(self, matrix, criteria):
            pass

        def _validate_filters(self, filters):
            pass

    tfm = FooFilter({"ZARAZA": 1})

    with pytest.raises(ValueError):
        tfm.transform(dm)


# =============================================================================
# FILTER
# =============================================================================


def test_Filter_criteria_is_not_str():

    with pytest.raises(ValueError):
        filters.Filter({1: lambda e: e > 1})


def test_Filter_filter_is_not_callable():

    with pytest.raises(ValueError):
        filters.Filter({"foo": 2})


def test_Filter():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 6, 28],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "AA", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.Filter(
        {
            "ROE": lambda e: e > 1,
            "RI": lambda e: e >= 28,
        }
    )

    result = tfm.transform(dm)
    assert result.equals(expected)


# =============================================================================
# ARITHMETIC FILTER
# =============================================================================


def test_SKCArithmeticFilterABC_not_implemented__filter():
    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    class FooFilter(filters.SKCArithmeticFilterABC):
        def _filter(self, arr, cond):
            return super()._filter(arr, cond)

    tfm = FooFilter({"ROE": 1})

    with pytest.raises(NotImplementedError):
        tfm.transform(dm)


def test_SKCArithmeticFilterABC_criteria_is_not_str():
    class FooFilter(filters.SKCArithmeticFilterABC):
        _filter = np.greater

    with pytest.raises(ValueError):
        FooFilter({1: 1})


def test_SKCArithmeticFilterABC_filter_is_not_a_number():
    class FooFilter(filters.SKCArithmeticFilterABC):
        _filter = np.greater

    with pytest.raises(ValueError):
        FooFilter({"foo": "nope"})


def test_FilterGT():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 6, 28],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "AA", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterGT({"ROE": 1, "RI": 27})

    result = tfm.transform(dm)
    assert result.equals(expected)


def test_FilterGE():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 6, 28],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "AA", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterGE({"ROE": 2, "RI": 28})

    result = tfm.transform(dm)
    assert result.equals(expected)


def test_FilterGE():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 6, 28],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "AA", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterGE({"ROE": 2, "RI": 28})

    result = tfm.transform(dm)
    assert result.equals(expected)


def test_FilterLT():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [5, 4, 26],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["JN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterLT({"ROE": 7, "CAP": 5})

    result = tfm.transform(dm)
    assert result.equals(expected)


def test_FilterLE():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [5, 4, 26],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["JN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterLE({"ROE": 5, "CAP": 4})

    result = tfm.transform(dm)
    assert result.equals(expected)


def test_FilterEQ():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [5, 4, 26],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["JN"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterEQ({"ROE": 5, "CAP": 4})

    result = tfm.transform(dm)
    assert result.equals(expected)


def test_FilterNE():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [1, 7, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "MM"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterNE({"ROE": 5, "CAP": 4})

    result = tfm.transform(dm)

    assert result.equals(expected)


# =============================================================================
# SET FILTER
# =============================================================================

def test_SKCSetFilterABC_not_implemented__set_filter():
    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    class FooFilter(filters.SKCSetFilterABC):
        def _set_filter(self, arr, cond):
            return super()._set_filter(arr, cond)

    tfm = FooFilter({"ROE": [1]})

    with pytest.raises(NotImplementedError):
        tfm.transform(dm)

def test_SKCSetFilterABC_criteria_is_not_str():
    class FooFilter(filters.SKCSetFilterABC):
        def _set_filter(self, arr, cond):
            pass

    with pytest.raises(ValueError):
        FooFilter({1: [1]})


def test_SKCSetFilterABC_filter_is_not_a_number():
    class FooFilter(filters.SKCSetFilterABC):
        def _set_filter(self, arr, cond):
            pass

    with pytest.raises(ValueError):
        FooFilter({"foo": 1})


def test_FilterIn():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [5, 4, 26],
            [1, 7, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["JN", "MM"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterIn({"ROE": [5, 1], "CAP": [4, 7]})

    result = tfm.transform(dm)

    assert result.equals(expected)


def test_FilterNotIn():

    dm = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "MM", "FN"],
        criteria=["ROE", "CAP", "RI"],
    )

    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE"],
        criteria=["ROE", "CAP", "RI"],
    )

    tfm = filters.FilterNotIn({"ROE": [5, 1], "CAP": [4, 7]})

    result = tfm.transform(dm)

    assert result.equals(expected)
