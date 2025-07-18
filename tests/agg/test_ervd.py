#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.ervd"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.ervd import ERVD
from skcriteria.preprocessing.scalers import SumScaler

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_ervd_invalid_metric():
    with pytest.raises(ValueError):
        ERVD(metric="foo")


def test_ervd_reference_points_none():
    alternatives_matrix = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    weights = np.array([0.5, 0.5])
    objectives = np.ones(2)

    dm = skcriteria.mkdm(
        matrix=alternatives_matrix,
        objectives=objectives,
        weights=weights,
    )

    ranker = ERVD()

    with pytest.raises(
        ValueError,
        match="Reference points must be provided for ERVD evaluation.",
    ):
        ranker.evaluate(dm, reference_points=None)


def test_ervd_reference_points_invalid_legth():
    alternatives_matrix = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    weights = np.array([0.5, 0.5])
    objectives = np.ones(2)
    reference_points = np.array([1, 2, 3])  # Invalid length
    dm = skcriteria.mkdm(
        matrix=alternatives_matrix,
        objectives=objectives,
        weights=weights,
    )

    ranker = ERVD()

    with pytest.raises(
        ValueError,
        match="Reference points must match the number of criteria in "
        "the decision matrix.",
    ):
        ranker.evaluate(dm, reference_points=reference_points)


def test_ERVD_shyur2015multiple():
    """
    Data from

    Shyur, H. J., Yin, L., Shih, H. S., & Cheng, C. B. (2015).
    A multiple criteria decision making method based on relative
    value distances.
    Foundations of Computing and Decision Sciences, 40(4), 299-315.
    """
    alternatives_matrix = np.array(
        [
            [80, 70, 87, 77, 76, 80, 75],
            [85, 65, 76, 80, 75, 65, 75],
            [78, 90, 72, 80, 85, 90, 85],
            [75, 84, 69, 85, 65, 65, 70],
            [84, 67, 60, 75, 85, 75, 80],
            [85, 78, 82, 81, 79, 80, 80],
            [77, 83, 74, 70, 71, 65, 70],
            [78, 82, 72, 80, 78, 70, 60],
            [85, 90, 80, 88, 90, 80, 85],
            [89, 75, 79, 67, 77, 70, 75],
            [65, 55, 68, 62, 70, 50, 60],
            [70, 64, 65, 65, 60, 60, 65],
            [95, 80, 70, 75, 70, 75, 75],
            [70, 80, 79, 80, 85, 80, 70],
            [60, 78, 87, 70, 66, 70, 65],
            [92, 85, 88, 90, 85, 90, 95],
            [86, 87, 80, 70, 72, 80, 85],
        ]
    )

    weights = np.array([0.066, 0.196, 0.066, 0.130, 0.130, 0.216, 0.196])
    assert sum(weights) == 1.0, "Weights must sum to 1."

    objectives = np.ones(7)
    reference_points = np.ones(7) * 80

    dm = skcriteria.mkdm(
        matrix=alternatives_matrix,
        objectives=objectives,
        weights=weights,
    )

    transformer = SumScaler(target="matrix")

    # Normalize the alternatives matrix
    n_dm = transformer.transform(dm)
    # Scale the reference points
    n_reference_points = reference_points / np.sum(alternatives_matrix, axis=0)

    ranker = ERVD()
    result: RankResult = ranker.evaluate(
        n_dm, reference_points=n_reference_points
    )

    expected = RankResult(
        method="ERVD",
        alternatives=[
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16",
        ],
        values=[7, 13, 3, 12, 9, 4, 14, 11, 2, 10, 17, 16, 8, 6, 15, 1, 5],
        extra={
            "similarity": [
                0.660,
                0.503,
                0.885,
                0.521,
                0.610,
                0.796,
                0.498,
                0.549,
                0.908,
                0.565,
                0.070,
                0.199,
                0.632,
                0.716,
                0.438,
                0.972,
                0.767,
            ],
            "s_plus": [
                0.027,
                0.040,
                0.009,
                0.038,
                0.031,
                0.016,
                0.040,
                0.036,
                0.007,
                0.035,
                0.075,
                0.064,
                0.030,
                0.023,
                0.045,
                0.002,
                0.019,
            ],
            "s_minus": [
                0.053,
                0.040,
                0.071,
                0.042,
                0.049,
                0.064,
                0.040,
                0.044,
                0.073,
                0.045,
                0.006,
                0.016,
                0.051,
                0.057,
                0.035,
                0.078,
                0.062,
            ],
        },
    )
    assert result.values_equals(expected)
    assert result.method == expected.method

    assert np.allclose(
        result.e_.similarity, expected.e_.similarity, atol=1.0e-3
    )
    assert np.allclose(result.e_.s_plus, expected.e_.s_plus, atol=1.0e-3)
    assert np.allclose(result.e_.s_minus, expected.e_.s_minus, atol=1.0e-3)


def test_decreasing_value_function():
    alternatives_matrix = np.array(
        [
            [5, 2, 7],
            [9, 3, 5],
            [7, 1, 8],
        ]
    )

    weights = np.array([1, 1, 1])
    objectives = [1, -1, 1]
    reference_points = np.ones(3) * 5

    transformer = SumScaler(target="matrix")

    dm = skcriteria.mkdm(
        matrix=alternatives_matrix,
        objectives=objectives,
        weights=weights,
    )
    # Normalize the alternatives matrix
    n_dm = transformer.transform(dm)

    # Scale the reference points
    n_reference_points = reference_points / np.sum(alternatives_matrix, axis=0)
    ranker = ERVD()
    result: RankResult = ranker.evaluate(
        n_dm, reference_points=n_reference_points
    )

    expected = RankResult(
        method="ERVD",
        alternatives=[
            "A0",
            "A1",
            "A2",
        ],
        values=[2, 3, 1],
        extra={
            "similarity": [0.398, 0.313, 0.856],
            "s_plus": [0.445, 0.507, 0.106],
            "s_minus": [0.294, 0.232, 0.634],
        },
    )

    assert result.values_equals(expected)
    assert np.allclose(result.e_.s_plus, expected.e_.s_plus, atol=1.0e-3)
    assert np.allclose(result.e_.s_minus, expected.e_.s_minus, atol=1.0e-3)
    assert np.allclose(
        result.e_.similarity, expected.e_.similarity, atol=1.0e-3
    )
