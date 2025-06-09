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
from skcriteria.agg.similarity import TOPSIS, VIKOR
from skcriteria.preprocessing.scalers import VectorScaler

# =============================================================================
# TOPSIS
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
#
# =============================================================================


@pytest.mark.xfail(
    reason="This divides by zero and VIKOR does not check for it"
)
def test_VIKOR_old():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, max, max],
    )

    expected = RankResult(
        "VIKOR",
        ["A0", "A1"],
        [2, 1],
        {
            "ideal": [1, 5, 6],
            "anti_ideal": [0, 0, 3],
            "similarity": [0.14639248, 0.85360752],
        },
    )

    ranker = VIKOR()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.ideal == expected.e_.ideal)
    assert np.allclose(result.e_.anti_ideal, expected.e_.anti_ideal)
    assert np.allclose(result.e_.similarity, expected.e_.similarity)


def test_VIKOR():
    dm = skcriteria.mkdm(
        matrix=[
            [5, 8, 4],
            [7, 6, 8],
            [8, 8, 6],
            [7, 4, 6],
        ],
        weights=[0.3, 0.4, 0.3],
        objectives=[max, max, max],
        alternatives=["A1", "A2", "A3", "A4"],
        criteria=["DUR", "CAP", "REL"],
    )

    expected = RankResult(
        "VIKOR",
        ["A1", "A2", "A3", "A4"],
        [3, 2, 1, 4],
        {
            "r_k": np.array([0.3, 0.2, 0.15, 0.4]),
            "s_k": np.array([0.6, 0.3, 0.15, 0.65]),
            "q_k": np.array([0.75, 0.25, 0.0, 1.0]),
            "acceptable_advantage": False,
            "acceptable_stability": True,
            "compromise_set": np.array([1, 2]),
        },
    )

    ranker = VIKOR()
    result = ranker.evaluate(dm)

    def DEBUG(result):
        print(result)
        for k, v in result.e_.items():
            print(f"{k:20}{v}")

    DEBUG(result)
    DEBUG(expected)

    diff = expected.diff(result)
    assert not diff.has_differences, diff


def test_VIKOR_tied():
    dm = skcriteria.mkdm(
        matrix=[
            [4, 4, 4],
            [7, 6, 8],
            [8, 8, 6],
            [6, 8, 8],
        ],
        weights=[0.3, 0.4, 0.3],
        objectives=[max, max, max],
        alternatives=["A1", "A2", "A3", "A4"],
        criteria=["DUR", "CAP", "REL"],
    )

    expected = RankResult(
        "VIKOR",
        ["A1", "A2", "A3", "A4"],
        [3, 2, 1, 1],
        {
            "r_k": np.array([0.4, 0.2, 0.15, 0.15]),
            "s_k": np.array([1.0, 0.275, 0.15, 0.15]),
            "q_k": np.array([1.0, 0.17352941, 0.0, 0.0]),
            "acceptable_advantage": False,
            "acceptable_stability": True,
            "compromise_set": np.array([1, 2, 3]),
        },
    )

    ranker = VIKOR()
    result = ranker.evaluate(dm)

    def DEBUG(result):
        print(result)
        for k, v in result.e_.items():
            print(f"{k:20}{v}")

    DEBUG(result)
    DEBUG(expected)

    diff = expected.diff(result)
    assert not diff.has_differences, diff


def test_VIKOR_invalid_v():
    with pytest.raises(ValueError):
        VIKOR(v=1.1)
    with pytest.raises(ValueError):
        VIKOR(v=-0.1)
