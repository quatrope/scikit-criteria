#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.similarity."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.similarity import TOPSIS
from skcriteria.agg.similarity import RIM
from skcriteria.preprocessing.scalers import VectorScaler

# =============================================================================
# TEST TOPSIS
# =============================================================================


def test_TOPSIS():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, max, max],
    )

    expected = RankResult(
        "TOPSIS",
        ["A0", "A1"],
        [2, 1],
        {
            "ideal": [1, 5, 6],
            "anti_ideal": [0, 0, 3],
            "similarity": [0.14639248, 0.85360752],
        },
    )

    ranker = TOPSIS()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.ideal == expected.e_.ideal)
    assert np.allclose(result.e_.anti_ideal, expected.e_.anti_ideal)
    assert np.allclose(result.e_.similarity, expected.e_.similarity)


def test_TOPSIS_invalid_metric():
    with pytest.raises(ValueError):
        TOPSIS(metric="foo")


def test_TOPSIS_minimize_warning():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = TOPSIS()

    with pytest.warns(UserWarning):
        ranker.evaluate(dm)


def test_TOPSIS_tzeng2011multiple():
    """
    Data from:
        Tzeng, G. H., & Huang, J. J. (2011).
        Multiple attribute decision making: methods and applications.
        CRC press.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [5, 8, 4],
            [7, 6, 8],
            [8, 8, 6],
            [7, 4, 6],
        ],
        objectives=[max, max, max],
        weights=[0.3, 0.4, 0.3],
    )

    transformer = VectorScaler(target="matrix")
    dm = transformer.transform(dm)

    expected = RankResult(
        "TOPSIS",
        ["A0", "A1", "A2", "A3"],
        [3, 2, 1, 4],
        {"similarity": [0.5037, 0.6581, 0.7482, 0.3340]},
    )

    ranker = TOPSIS()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(
        result.e_.similarity, expected.e_.similarity, atol=1.0e-4
    )


# =============================================================================
# TEST RIM
# =============================================================================


def test_RIM():
    # Real example taken from the RIM paper

    matrix = [
        [30, 0, 2, 3, 3, 2],  # A
        [40, 9, 1, 3, 2, 2],  # B
        [25, 0, 3, 1, 3, 2],  # C
        [27, 0, 5, 3, 3, 1],  # D
        [45, 15, 2, 2, 3, 4],  # E
    ]

    weights = [0.2262, 0.2143, 0.1786, 0.1429, 0.1190, 0.1190]

    ranges = [
        (23, 60),
        (0, 15),
        (0, 10),
        (1, 3),
        (1, 3),
        (1, 5),
    ]

    ref_intervals = [
        (30, 35),
        (10, 15),
        (0, 0),
        (3, 3),
        (3, 3),
        (4, 5),
    ]

    alternatives = ["A", "B", "C", "D", "E"]

    dm = skcriteria.mkdm(
        matrix=matrix,
        weights=weights,
        ref_ideals=ref_intervals,
        ranges=ranges,
        alternatives=alternatives,
    )

    result = RIM().evaluate(dm)

    expected = RankResult(
        "RIM",
        ["A", "B", "E", "D", "C"],
        [1, 2, 3, 4, 5],
        {
            "similarity": [0.58663, 0.51883, 0.49475, 0.17556, 0.10031],
        },
    )

    assert isinstance(result, RankResult)
    assert result.shape == (5,)
    assert result.values_equals(expected)


def test_RIM_invalid_values():
    # Error when entering values ​​outside the allowed ranges

    dm = skcriteria.mkdm(
        matrix=[[42]],
        weights=[1],
        ref_ideals=[(1, 2)],
        ranges=[(1, 2)],
        alternatives=["A"],
    )

    with pytest.raises(ValueError):
        RIM().evaluate(dm)
