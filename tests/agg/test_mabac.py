#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Test suite for MABAC method."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria import DecisionMatrix
from skcriteria.agg import MABAC
from skcriteria.agg import RankResult
from skcriteria.preprocessing.invert_objectives import MinMaxInverter


# =============================================================================
# TESTS
# =============================================================================


def test_MABAC_dragan_2014():
    """Data from
    The selection of transport and handling resources in logistics centers
    using Multi-Attributive Border Approximation area Comparison
    (MABAC)
    2014 Dragan Pamučar and Goran Ćirović

    """

    # Simple 3x3 decision matrix
    forklift_matrix = np.array(
        [
            [22600, 3800, 2, 5, 1.06, 3.00, 3.5, 2.8, 24.5, 6.5],  # Forklift 1
            [19500, 4200, 3, 2, 0.95, 3.00, 3.4, 2.2, 24.0, 7.0],  # Forklift 2
            [21700, 4000, 1, 3, 1.25, 3.20, 3.3, 2.5, 24.5, 7.3],  # Forklift 3
            [
                20600,
                3800,
                2,
                5,
                1.05,
                3.25,
                3.2,
                2.0,
                22.5,
                11.0,
            ],  # Forklift 4
            [22500, 3800, 4, 3, 1.35, 3.20, 3.7, 2.1, 23.0, 6.3],  # Forklift 5
            [23250, 4210, 3, 5, 1.45, 3.60, 3.5, 2.8, 23.5, 7.0],  # Forklift 6
            [20300, 3850, 2, 5, 0.90, 3.25, 3.0, 2.6, 21.5, 6.0],  # Forklift 7
        ]
    )

    objectives = np.array(
        [-1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
    )  # Minimize costs and maximize performance

    weights = np.array(
        [
            0.146,  # C1
            0.144,  # C2
            0.119,  # C3
            0.121,  # C4
            0.115,  # C5
            0.101,  # C6
            0.088,  # C7
            0.068,  # C8
            0.050,  # C9
            0.048,  # C10
        ]
    )

    expected_baa = [
        0.2086,
        0.1885,
        0.1720,
        0.1952,
        0.1740,
        0.1625,
        0.1319,
        0.1010,
        0.0789,
        0.0590,
    ]

    expected_score = [0.0826, 0.2183, -0.0488, 0.0246, -0.0704, 0.0465, 0.0464]

    expected_rank = [2, 1, 6, 5, 7, 3, 4]

    dm = DecisionMatrix.from_mcda_data(
        forklift_matrix,
        objectives,
        weights=weights,
        alternatives=[f"Forklift {i}" for i in range(1, 8)],
        criteria=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
    )

    expected = RankResult(
        "MABAC",
        [f"Forklift {i}" for i in range(1, 8)],
        expected_rank,
        {"border_approximation_area": expected_baa, "score": expected_score},
    )

    rdm = MinMaxInverter().transform(dm)
    result = MABAC().evaluate(rdm)
    baa_result = result.extra_["border_approximation_area"]
    score_result = result.extra_["score"]

    assert result.values_equals(expected)
    assert np.allclose(baa_result, expected_baa, atol=1.0e-3)
    assert np.allclose(score_result, expected_score, atol=1.0e-3)


def test_MABAC_minimize_value_error():
    """Test that MABAC raises an error for minimization objectives."""

    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    objectives = np.array([1, -1, 1])
    weights = np.array([0.3, 0.4, 0.3])

    dm = DecisionMatrix.from_mcda_data(
        matrix=matrix, objectives=objectives, weights=weights
    )

    mabac = MABAC()
    with pytest.raises(ValueError):
        mabac.evaluate(dm)
