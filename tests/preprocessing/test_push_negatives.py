#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.push_negatives"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.preprocessing.push_negatives import (
    PushNegatives,
    push_negatives,
)


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_PushNegatives_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, -2, 3], [-1, 5, -60]],
        objectives=[min, max, min],
        weights=[1, 2, -1],
    )

    expected = skcriteria.mkdm(
        matrix=[[61, 58, 63], [59, 65, 0]],
        objectives=[min, max, min],
        weights=[1, 2, -1],
    )

    scaler = PushNegatives(target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_PushNegatives_simple_matrix_ge0():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 1],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 1],
    )

    scaler = PushNegatives(target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_PushNegatives_simple_weights():
    dm = skcriteria.mkdm(
        matrix=[[1, -2, 3], [-1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, -1],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, -2, 3], [-1, 5, 6]],
        objectives=[min, max, min],
        weights=[2, 3, 0],
    )

    scaler = PushNegatives(target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_PushNegatives_simple_weights_ge0():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 1],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 1],
    )

    scaler = PushNegatives(target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_PushNegatives_simple_both():
    dm = skcriteria.mkdm(
        matrix=[[1, -2, 3], [-1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, -1],
    )

    expected = skcriteria.mkdm(
        matrix=[[3, 0, 5], [1, 7, 8]],
        objectives=[min, max, min],
        weights=[2, 3, 0],
    )

    scaler = PushNegatives(target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_PushNegatives_simple_both_ge0():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 1],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 1],
    )

    scaler = PushNegatives(target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def _test_PushNegatives_no_change_original_dm():
    dm = skcriteria.mkdm(
        matrix=[[-1, 0, 3], [0, -5, 6]],
        objectives=[min, max, min],
        weights=[1, -2, 0],
    )

    expected = dm.copy()

    tfm = PushNegatives(target="both")
    dmt = tfm.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )


def test_push_negatives_deprecated_call():
    with pytest.deprecated_call():
        result = push_negatives([-11, 2, 3], None)
    np.testing.assert_array_equal(result, [0, 2 + 11, 3 + 11])
