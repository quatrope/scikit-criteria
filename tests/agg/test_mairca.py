#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.mairca"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria import mkdm, testing
from skcriteria.agg import RankResult
from skcriteria.agg.mairca import MAIRCA

# =============================================================================
# TEST CLASSES
# =============================================================================


@pytest.fixture
def ljubomir2016combination_matrix():
    """
    Data From:
        Ljubomir Gigović, Dragan Pamučar, Zoran Bajić and Milić Milićević.
        The Combination of Expert Judgment and GIS-MAIRCA Analysis for the
        Selection of Sites for Ammunition Depots.
        Sustainability 2016, https://www.mdpi.com/2071-1050/8/4/372
    """
    matrix = np.array(
        [
            [3.828, 5.000, 3.720, 2.723, 4.255],
            [4.675, 5.000, 3.000, 3.452, 2.587],
            [4.515, 4.836, 3.289, 3.491, 4.069],
            [4.421, 5.000, 3.555, 2.839, 4.397],
            [4.717, 5.000, 3.430, 4.401, 1.000],
            [4.695, 5.000, 3.925, 3.847, 1.000],
            [4.688, 5.000, 2.000, 4.99, 1.000],
            [4.688, 5.000, 3.971, 4.001, 1.000],
        ]
    )
    objectives = np.array([min, max, min, min, min])
    weights = np.array([0.2016, 0.2304, 0.2232, 0.1912, 0.1536])
    return mkdm(matrix=matrix, objectives=objectives, weights=weights)


@pytest.fixture
def ngoctien2024application_matrix():
    """
    Data From:
        Ngoc-Tien Tran.
        APPLICATION OF THE MULTI-CRITERIA ANALYSIS METHOD MAIRCA, SPOTIS, COMET
        FOR THE OPTIMISATION OF SUSTAINABLE ELECTRICITY TECHNOLOGY DEVELOPMENT.
        School of Mechanical and Automotive Engineering, Hanoi University of
        Industry, Hanoi, Vietnam, 2024.
        https://journal.eu-jr.24eu/engineering/article/view/3133
    """
    matrix = np.array(
        [
            [0.19, 0.013, 2.653, 0.015, 0.1452, 0.001, 0.9],
            [2.39, 0.208, 7.194, 0.213, 0.0017, 0.132, 0.85],
            [1.548, 0.751, 3.203, 0.186, 0.0012, 0.157, 0.85],
            [0.198, 0.013, 7.229, 0.016, 0.0001, 0.001, 0.8],
            [0.142, 0.009, 4.519, 0.011, 0.0001, 0.001, 0.8],
            [0.142, 0.01, 6.019, 0.007, 0.0004, 0.001, 0.29],
            [0.173, 0.007, 6.143, 0.006, 0.0022, 0.001, 0.5],
            [0.479, 0.056, 25.14, 0.032, 0.0028, 0.001, 0.15],
            [1.082, 0.108, 20.829, 0.064, 0.0002, 0.001, 0.15],
            [1.406, 0.674, 0.945, 0.167, 0.001, 0.157, 0.85],
        ]
    )
    objectives = np.array([min, min, min, min, min, min, max])
    weights = np.array([0.1, 0.117, 0.215, 0.107, 0.172, 0.126, 0.163])
    return mkdm(matrix=matrix, objectives=objectives, weights=weights)


def test_MAIRCA_with_equal_P_ai_ljubomir2016combination(
    ljubomir2016combination_matrix,
):
    P_ai = np.array([0.125] * 8)

    Q_expected = np.array(
        [0.0427, 0.0548, 0.0919, 0.0592, 0.0631, 0.0637, 0.0483, 0.0658]
    )
    ranking_expected = np.array([1, 3, 8, 4, 5, 6, 2, 7])
    alternatives_expected = [
        f"A{i}" for i in range(len(ljubomir2016combination_matrix.matrix))
    ]

    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    mairca = MAIRCA()
    result = mairca.evaluate(ljubomir2016combination_matrix, P_ai=P_ai)

    testing.assert_result_equals(result, expected, atol=1e-4)


def test_MAIRCA_without_P_ai_ljubomir2016combination(
    ljubomir2016combination_matrix,
):
    Q_expected = np.array(
        [0.0427, 0.0548, 0.0919, 0.0592, 0.0631, 0.0637, 0.0483, 0.0658]
    )
    ranking_expected = np.array([1, 3, 8, 4, 5, 6, 2, 7])
    alternatives_expected = [
        f"A{i}" for i in range(len(ljubomir2016combination_matrix.matrix))
    ]

    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    mairca = MAIRCA()
    result = mairca.evaluate(ljubomir2016combination_matrix)

    testing.assert_result_equals(result, expected, atol=1e-4)


def test_MAIRCA_with_custom_P_ai_ljubomir2016combination(
    ljubomir2016combination_matrix,
):
    P_ai = np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])

    Q_expected = np.array(
        [
            0.06839111,
            0.06578403,
            0.11035555,
            0.04739504,
            0.05050595,
            0.05094007,
            0.03862236,
            0.05260108,
        ]
    )
    ranking_expected = np.array([7, 6, 8, 2, 3, 4, 1, 5])
    alternatives_expected = [
        f"A{i}" for i in range(len(ljubomir2016combination_matrix.matrix))
    ]
    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    mairca_dm = MAIRCA()
    result = mairca_dm.evaluate(ljubomir2016combination_matrix, P_ai=P_ai)

    testing.assert_result_equals(result, expected, atol=1e-4)


def test_MAIRCA_ngoctien2024application(ngoctien2024application_matrix):

    Q_expected = np.array(
        [
            0.019491,
            0.041271,
            0.043082,
            0.008618,
            0.005639,
            0.017901,
            0.013699,
            0.041734,
            0.042749,
            0.03822745,
        ]
    )
    ranking_expected = np.array([5, 7, 10, 2, 1, 4, 3, 8, 9, 6])
    alternatives_expected = [
        f"A{i}" for i in range(len(ngoctien2024application_matrix.matrix))
    ]

    extra = {"values": Q_expected}

    expected = RankResult(
        "MAIRCA",
        alternatives=alternatives_expected,
        values=ranking_expected,
        extra=extra,
    )

    mairca = MAIRCA()
    result = mairca.evaluate(ngoctien2024application_matrix)

    testing.assert_result_equals(result, expected, atol=1e-4)


def test_MAIRCA_mismatched_P_ai_length(ljubomir2016combination_matrix):
    P_ai = np.array([0.2] * 5)

    with pytest.raises(
        ValueError, match="Length of P_ai must match number of alternatives"
    ):
        mairca_dm = MAIRCA()
        mairca_dm.evaluate(ljubomir2016combination_matrix, P_ai=P_ai)


def test_MAIRCA_invalid_P_ai_sum(ljubomir2016combination_matrix):
    P_ai = np.array([0.2] * 8)

    with pytest.raises(ValueError, match="Sum of P_ai must be 1"):
        mairca = MAIRCA()
        mairca.evaluate(ljubomir2016combination_matrix, P_ai=P_ai)


def test_MAIRCA_negative_P_ai(ljubomir2016combination_matrix):
    P_ai = np.array([0.2] * 7 + [-0.4])

    with pytest.raises(ValueError, match="P_ai must be non-negative"):
        mairca = MAIRCA()
        mairca.evaluate(ljubomir2016combination_matrix, P_ai=P_ai)


def test_MAIRCA_zero_P_ai(ljubomir2016combination_matrix):
    P_ai = np.array([0.2] * 5 + [0.0] * 3)

    mairca = MAIRCA()
    result = mairca.evaluate(ljubomir2016combination_matrix, P_ai=P_ai)
    assert len(result.e_["values"]) == len(
        ljubomir2016combination_matrix.matrix
    )


def test_MAIRCA_zero_in_matrix():
    dm = mkdm([[0]], [max])
    dec = MAIRCA()
    with pytest.raises(ValueError):
        dec.evaluate(dm)


def test_MAIRCA_equal_values_in_criterion():
    dm = mkdm(
        matrix=[
            [2.0, 1.0],
            [2.0, 3.0],
        ],
        objectives=[max, max],
        weights=[0.6, 0.4],
    )
    mairca = MAIRCA()
    Q_i = np.array([0.2, 0])

    expected = RankResult(
        "MAIRCA",
        alternatives=["A0", "A1"],
        values=[2, 1],
        extra={"values": Q_i},
    )

    result = mairca.evaluate(dm)

    testing.assert_result_equals(result, expected, atol=1e-4)
