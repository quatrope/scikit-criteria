#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.norm.sum_normalizer

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.norm import sum_normalizer


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_bad_SumNormalizer_normalize_for():
    with pytest.raises(ValueError):
        sum_normalizer.SumNormalizer(normalize_for="mtx")


def test_SumNormalizer_simple_matrix():

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

    normalizer = sum_normalizer.SumNormalizer(normalize_for="matrix")

    result = normalizer.normalize(dm)

    assert result == expected


def test_SumNormalizer_matrix(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix
        / np.sum(dm.matrix, axis=0, keepdims=True, dtype=float),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    normalizer = sum_normalizer.SumNormalizer(normalize_for="matrix")
    result = normalizer.normalize(dm)

    assert result == expected


def test_SumNormalizer_simple_weights():

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

    normalizer = sum_normalizer.SumNormalizer(normalize_for="weights")

    result = normalizer.normalize(dm)

    assert result == expected


def test_SumNormalizer_weights(decision_matrix):

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
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    normalizer = sum_normalizer.SumNormalizer(normalize_for="weights")
    result = normalizer.normalize(dm)

    assert result == expected


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_sum_norm_mtx(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nweights = sum_normalizer.sum_norm(dm.weights, axis=0)
    expected = dm.weights / np.sum(dm.weights)

    assert np.all(nweights == expected)


def test_sum_norm_weights(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nmtx = sum_normalizer.sum_norm(dm.matrix, axis=0)
    expected = dm.matrix / np.sum(
        dm.matrix, axis=0, keepdims=True, dtype=float
    )

    assert np.all(nmtx == expected)
