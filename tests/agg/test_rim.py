#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.rim"""

# =============================================================================
# IMPORTS
# =============================================================================

import re

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.rim import RIM, _rim_normalize


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

    ref_ideals = [
        (30, 35),
        (10, 15),
        (0, 0),
        (3, 3),
        (3, 3),
        (4, 5),
    ]

    alternatives = ["A", "B", "C", "D", "E"]
    objectives = [max] * 6

    dm = skcriteria.mkdm(
        matrix=matrix,
        weights=weights,
        objectives=objectives,
        alternatives=alternatives,
    )

    rim = RIM()
    result = rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)

    i_plus_expected = np.array([0.23129, 0.11251, 0.31877, 0.27831, 0.12070])
    i_minus_expected = np.array([0.32823, 0.34831, 0.18852, 0.24344, 0.34378])
    score_expected = np.array([0.58663, 0.75560, 0.37165, 0.46671, 0.73988])

    # the calculation on paper is incorrect
    # i_minus_paper = np.array([0.32823, 0.12132, 0.03554, 0.05926, 0.11819])
    # score_expected = np.array([0.58663, 0.51883, 0.10031, 0.17556, 0.49475])

    norm_matrix_expected = np.array(
        [
            [1.0, 0.0, 0.8, 1.0, 1.0, 0.3333],  # A
            [0.8, 0.9, 0.9, 1.0, 0.5, 0.3333],  # B
            [0.2857, 0.0, 0.7, 0.0, 1.0, 0.3333],  # C
            [0.5714, 0.0, 0.5, 1.0, 1.0, 0.0],  # D
            [0.6, 1.0, 0.8, 0.5, 1.0, 1.0],  # E
        ]
    )

    weighted_matrix_expected = np.array(
        [
            [0.22620, 0.00000, 0.14288, 0.14290, 0.11900, 0.03967],
            [0.18096, 0.19287, 0.16074, 0.14290, 0.05950, 0.03967],
            [0.06463, 0.00000, 0.12502, 0.00000, 0.11900, 0.03967],
            [0.12926, 0.00000, 0.08930, 0.14290, 0.11900, 0.00000],
            [0.13572, 0.21430, 0.14288, 0.07145, 0.11900, 0.11900],
        ]
    )

    expected = RankResult(
        "RIM",
        ["A", "B", "C", "D", "E"],
        [3, 1, 5, 4, 2],
        {
            "score": score_expected,
            "norm_matrix": norm_matrix_expected,
            "weighted_matrix": weighted_matrix_expected,
            "i_plus": i_plus_expected,
            "i_minus": i_minus_expected,
        },
    )

    assert isinstance(result, RankResult)
    assert result.shape == (5,)

    assert np.allclose(
        result.e_["norm_matrix"], expected.e_["norm_matrix"], atol=1e-3
    )
    assert np.allclose(
        result.e_["weighted_matrix"], expected.e_["weighted_matrix"], atol=1e-4
    )
    assert np.allclose(result.e_["i_plus"], expected.e_["i_plus"], atol=1e-4)
    assert np.allclose(result.e_["i_minus"], expected.e_["i_minus"], atol=1e-4)
    assert np.allclose(result.e_["score"], expected.e_["score"], atol=1e-3)

    assert result.rank_.tolist() == expected.rank_.tolist()


def test_RIM_invalid_values():
    # Error when entering values ​​outside the allowed ranges

    matrix = [[42]]
    weights = [1]
    objectives = [max]
    ranges = [(1, 2)]
    ref_ideals = [(1, 2)]

    dm = skcriteria.mkdm(
        matrix=matrix,
        weights=weights,
        objectives=objectives,
        alternatives=["A"],
    )

    rim = RIM()

    with pytest.raises(
        ValueError,
        match=re.escape("Value 42 outside normalization range (1, 2)"),
    ):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)


def test_RIM_invalid_ref_ideals_length():
    matrix = [[1, 2]]
    weights = [1, 1]
    objectives = [max, max]
    ranges = [(0, 10), (0, 10)]
    ref_ideals = [(1, 2)]  # miss one ideal

    dm = skcriteria.mkdm(matrix=matrix, weights=weights, objectives=objectives)
    rim = RIM()

    with pytest.raises(ValueError, match="ref_ideals length"):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)


def test_RIM_invalid_ranges_length():
    matrix = [[1, 2]]
    weights = [1, 1]
    objectives = [max, max]
    ranges = [(0, 10)]  # miss one range
    ref_ideals = [(1, 2), (3, 4)]

    dm = skcriteria.mkdm(matrix=matrix, weights=weights, objectives=objectives)
    rim = RIM()

    with pytest.raises(
        ValueError, match="Ranges length must match number of criteria."
    ):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)


def test_RIM_ideal_outside_range():
    matrix = [[1, 2]]
    weights = [1, 1]
    objectives = [max, max]
    ranges = [(0, 10), (0, 5)]
    ref_ideals = [(1, 2), (6, 7)]  # outside range

    dm = skcriteria.mkdm(matrix=matrix, weights=weights, objectives=objectives)
    rim = RIM()

    with pytest.raises(ValueError, match="must be within ranges"):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)


def test_RIM_invalid_ranges_shape():
    matrix = [[1, 2]]
    weights = [1, 1]
    objectives = [max, max]

    # Correct length (2 elements), but incorrect shape (flat list)
    ranges = [0, 10]  # Should be [(0, 10), (0, 10)]
    ref_ideals = [(1, 2), (1, 2)]

    dm = skcriteria.mkdm(matrix=matrix, weights=weights, objectives=objectives)
    rim = RIM()

    with pytest.raises(
        TypeError, match="Each range must be a tuple or list of length 2."
    ):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)


def test_RIM_invalid_ref_ideal_shape():
    matrix = [[1, 2]]
    weights = [1, 1]
    objectives = [max, max]

    # Correct length (2 elements), but incorrect shape (flat list)
    ranges = [(0, 10), (0, 10)]
    ref_ideals = [1, 2]  # Should be [(1, 2), (1, 2)]

    dm = skcriteria.mkdm(matrix=matrix, weights=weights, objectives=objectives)
    rim = RIM()

    with pytest.raises(
        TypeError, match="Each ref_ideal must be a tuple or list of length 2."
    ):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)


def test_RIM_default_ref_and_ranges():
    matrix = [[1, 9], [10, 2]]
    weights = [0.5, 0.5]
    objectives = [max, min]
    alternatives = ["A", "B"]

    dm = skcriteria.mkdm(
        matrix=matrix,
        weights=weights,
        objectives=objectives,
        alternatives=alternatives,
    )

    rim = RIM()
    result = rim.evaluate(dm)
    expected_ranking = [2, 1]
    assert result.rank_.tolist() == expected_ranking

    # Validate defaults reference ideals and ranges
    expected_ref_ideals = [[10, 10], [2, 2]]
    expected_ranges = [[1, 10], [2, 9]]

    assert np.allclose(result.e_["ref_ideals"], expected_ref_ideals)
    assert np.allclose(result.e_["ranges"], expected_ranges)


def test_rim_normalize():

    # Define test cases with expected results
    cases = [
        # Inside the ideal interval
        {
            "value": 5,
            "value_range": (0, 10),
            "ref_ideal": (4, 6),
            "expected": 1.0,
        },
        # Before the ideal interval
        {
            "value": 3,
            "value_range": (0, 10),
            "ref_ideal": (4, 6),
            "expected": 0.75,
        },
        # After the ideal interval
        {
            "value": 7,
            "value_range": (0, 10),
            "ref_ideal": (4, 6),
            "expected": 0.75,
        },
        # Zero-length ideal (value exactly in it)
        {
            "value": 4,
            "value_range": (0, 10),
            "ref_ideal": (4, 4),
            "expected": 1.0,
        },
    ]

    for case in cases:
        result = _rim_normalize(
            case["value"],
            case["value_range"],
            case["ref_ideal"],
        )
        assert np.isclose(
            result, case["expected"], atol=1e-2
        ), f"Failed case: {case}"
