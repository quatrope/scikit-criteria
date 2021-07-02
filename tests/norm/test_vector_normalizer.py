#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.norm.vector_normalizer

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.norm import vector_normalizer


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_VectorNormalizer_simple_matrix():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [1 / 4.123105626, 2 / 5.385164807, 3 / 6.708203932],
            [4 / 4.123105626, 5 / 5.385164807, 6 / 6.708203932],
        ],
        objectives=[min, max, min],
        weights=[1, 2, 3],
        dtypes=[float, float, float],
    )

    normalizer = vector_normalizer.VectorNormalizer(normalize_for="matrix")

    result = normalizer.normalize(dm)

    assert result.aequals(expected)


def test_VectorNormalizer_matrix(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix / np.sqrt(np.sum(np.power(dm.matrix, 2), axis=0)),
        objectives=dm.objectives,
        weights=dm.weights,
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    normalizer = vector_normalizer.VectorNormalizer(normalize_for="matrix")
    result = normalizer.normalize(dm)

    assert result.aequals(expected)


def test_VectorNormalizer_simple_weights():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1 / 3.741657387, 2 / 3.741657387, 3 / 3.741657387],
        dtypes=[int, int, int],
    )

    normalizer = vector_normalizer.VectorNormalizer(normalize_for="weights")

    result = normalizer.normalize(dm)

    assert result.aequals(expected)


def test_VectorNormalizer_weights(decision_matrix):

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
        weights=dm.weights / np.sqrt(np.sum(np.power(dm.weights, 2))),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    normalizer = vector_normalizer.VectorNormalizer(normalize_for="weights")
    result = normalizer.normalize(dm)

    assert result.aequals(expected)


def test_VectorNormalizer_simple_both():

    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 3],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [1 / 4.123105626, 2 / 5.385164807, 3 / 6.708203932],
            [4 / 4.123105626, 5 / 5.385164807, 6 / 6.708203932],
        ],
        objectives=[min, max, min],
        weights=[1 / 3.741657387, 2 / 3.741657387, 3 / 3.741657387],
        dtypes=[float, float, float],
    )

    normalizer = vector_normalizer.VectorNormalizer(normalize_for="both")

    result = normalizer.normalize(dm)

    assert result.aequals(expected)


def test_VectorNormalizer_both(decision_matrix):

    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix / np.sqrt(np.sum(np.power(dm.matrix, 2), axis=0)),
        objectives=dm.objectives,
        weights=dm.weights / np.sqrt(np.sum(np.power(dm.weights, 2))),
        anames=dm.anames,
        cnames=dm.cnames,
        dtypes=dm.dtypes,
    )

    normalizer = vector_normalizer.VectorNormalizer(normalize_for="both")
    result = normalizer.normalize(dm)

    assert result.aequals(expected)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_vector_norm_mtx(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nweights = vector_normalizer.vector_norm(dm.weights, axis=0)
    expected = dm.weights / np.sqrt(np.sum(np.power(dm.weights, 2)))

    assert np.all(nweights == expected)


def test_vector_norm_weights(decision_matrix):

    dm = decision_matrix(
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=1.0,
    )

    nmtx = vector_normalizer.vector_norm(dm.matrix, axis=0)
    expected = dm.matrix / np.sqrt(np.sum(np.power(dm.matrix, 2), axis=0))

    assert np.all(nmtx == expected)
