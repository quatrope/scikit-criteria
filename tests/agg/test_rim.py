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


import pytest
import numpy as np
import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.rim import RIM


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
    i_minus_expected = np.array([0.32823, 0.12132, 0.03554, 0.05926, 0.11819])
    score_expected = np.array([0.58663, 0.51883, 0.10031, 0.17556, 0.49475])

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
        ["A", "B", "E", "D", "C"],
        [1, 2, 3, 4, 5],
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
    np.testing.assert_allclose(
        result.e_["norm_matrix"], expected.e_["norm_matrix"], atol=1e-4
    )
    np.testing.assert_allclose(
        result.e_["weighted_matrix"], expected.e_["weighted_matrix"], atol=1e-4
    )
    np.testing.assert_allclose(
        result.e_["i_plus"], expected.e_["i_plus"], atol=1e-5
    )
    np.testing.assert_allclose(
        result.e_["i_minus"], expected.e_["i_minus"], atol=1e-5
    )
    assert np.allclose(result.e_["score"], expected.e_["score"], atol=1e-5)

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

    with pytest.raises(ValueError, match="Outside the accepted range"):
        rim.evaluate(dm, ref_ideals=ref_ideals, ranges=ranges)
