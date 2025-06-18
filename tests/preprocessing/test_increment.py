#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.add_value_to_zero"""


# =============================================================================
# IMPORTS
# =============================================================================

import skcriteria
from skcriteria.preprocessing.increment import AddValueToZero


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_AddValueToZero_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1.0, 0.0, 3], [0.0, 5.0, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skcriteria.mkdm(
        matrix=[[1.5, 0.5, 3.0], [0.5, 5.5, 6.0]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    scaler = AddValueToZero(value=0.5, target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddValueToZero_simple_matrix_gt0():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    scaler = AddValueToZero(value=0.5, target="matrix")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddValueToZero_simple_weights():
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

    scaler = AddValueToZero(value=0.5, target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddValueToZero_simple_weights_gt0():
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

    scaler = AddValueToZero(value=0.5, target="weights")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddValueToZero_simple_both():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skcriteria.mkdm(
        matrix=[[1.5, 0.5, 3.0], [0.5, 5.5, 6.0]],
        objectives=[min, max, min],
        weights=[1.5, 2.5, 0.5],
    )

    scaler = AddValueToZero(value=0.5, target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddValueToZero_simple_both_gt0():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    scaler = AddValueToZero(value=0.5, target="both")

    result = scaler.transform(dm)

    assert result.equals(expected)


def test_AddValueToZero_no_change_original_dm():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = dm.copy()

    tfm = AddValueToZero(value=0.5, target="both")
    dmt = tfm.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )
