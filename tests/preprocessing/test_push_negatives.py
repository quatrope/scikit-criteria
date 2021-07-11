#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.push_negatives

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.preprocessing import PushNegatives, push_negatives


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_PushNegatives_simple_matrix():

    dm = skcriteria.mkdm(
        matrix=[[1, -2, 3], [-1, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, -1],
    )

    expected = skcriteria.mkdm(
        matrix=[[2, 0, 3], [0, 7, 6]],
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
        matrix=[[2, 0, 3], [0, 7, 6]],
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


def test_PushNegatives_no_change_original_dm():

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


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_push_negatives_weights():

    weights = [1, 2, -1]

    nweights = push_negatives(weights, axis=0)
    expected = [2, 3, 0]

    assert np.all(nweights == expected)


def test_push_negatives_weights_ge0():

    weights = [1, 2, 3]

    nweights = push_negatives(weights, axis=0)
    expected = [1, 2, 3]

    assert np.all(nweights == expected)


def test_push_negatives_matrix():

    matrix = [[1, -2, 3], [-1, 5, 6]]

    nmtx = push_negatives(matrix, axis=0)
    expected = [[2, 0, 3], [0, 7, 6]]

    assert np.all(nmtx == expected)


def test_push_negatives_matrix_ge0():

    matrix = [[1, 2, 3], [1, 5, 6]]

    nmtx = push_negatives(matrix, axis=0)
    expected = [[1, 2, 3], [1, 5, 6]]

    assert np.all(nmtx == expected)
