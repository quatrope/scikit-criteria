#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocess.add_eps_to_zero

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.preprocess import AddEpsToZero, add_eps_to_zero


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_AddEpsToZero_simple_matrix():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skcriteria.mkdm(
        matrix=[[1.5, 0.5, 3], [0.5, 5.5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    scaler = AddEpsToZero(eps=0.5, target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddEpsToZero_simple_matrix_gt0():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    scaler = AddEpsToZero(eps=0.5, target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddEpsToZero_simple_weights():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1.5, 2.5, 0.5],
    )

    scaler = AddEpsToZero(eps=0.5, target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddEpsToZero_simple_weights_gt0():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    scaler = AddEpsToZero(eps=0.5, target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddEpsToZero_simple_both():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skcriteria.mkdm(
        matrix=[[1.5, 0.5, 3], [0.5, 5.5, 6]],
        objectives=[min, max, min],
        weights=[1.5, 2.5, 0.5],
    )

    scaler = AddEpsToZero(eps=0.5, target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddEpsToZero_simple_both_gt0():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    scaler = AddEpsToZero(eps=0.5, target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


@pytest.mark.parametrize(
    "arr, expected",
    [([1, 2, 0], [1.5, 2.5, 0.5]), ([1, 2, 3], [1, 2, 3])],
)
def test_add_eps_to_zero_1D(arr, expected):
    result = add_eps_to_zero(arr, eps=0.5)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "arr, axis, expected",
    [
        ([[0, 0, 3], [4, 5, 6]], None, [[0.5, 0.5, 3.5], [4.5, 5.5, 6.5]]),
        ([[1, 2, 3], [4, 5, 6]], None, [[1, 2, 3], [4, 5, 6]]),
        ([[0, 0, 3], [4, 5, 6]], 0, [[0.5, 0.5, 3], [4.5, 5.5, 6]]),
        ([[1, 2, 3], [4, 5, 6]], 0, [[1, 2, 3], [4, 5, 6]]),
        ([[0, 0, 3], [4, 5, 6]], 1, [[0.5, 0.5, 3.5], [4, 5, 6]]),
        ([[1, 2, 3], [4, 5, 6]], 1, [[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_add_eps_to_zero_2D_columns(arr, axis, expected):
    arr = add_eps_to_zero(arr, eps=0.5, axis=axis)
    assert np.all(arr == expected)
