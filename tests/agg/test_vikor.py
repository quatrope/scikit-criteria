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
from skcriteria.agg.vikor import VIKOR

# =============================================================================
# VIKOR
# =============================================================================


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param([[1, 0, 3], [5, 0, 6]], id="criterion_all_equals"),
        pytest.param([[1, 0, 3], [0, 5, 6]], id="regret_all_equals"),
    ],
)
def test_VIKOR_warn_zerodiv(matrix):
    """Test that VIKOR warns when a scaling causes a divison by zero.

    This can happen when a criterion has the same value across all alternatives
    or when R_k (regret, or max distance) or S_k are equal for all alternatives
    and it results in the criterion (or R_k or S_k) being ignored.
    """
    dm = skcriteria.mkdm(matrix=matrix, objectives=[max, max, max])

    expected = RankResult("VIKOR", ["A0", "A1"], [1, 1], {})

    ranker = VIKOR()
    with pytest.warns(UserWarning):
        result = ranker.evaluate(dm)
    assert np.all(result.alternatives == expected.alternatives)
    assert result.method == expected.method


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
        [2, 1, 1, 3],
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
    assert expected.aequals(result)


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
        [2, 1, 1, 1],
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
    assert expected.aequals(result)


def test_VIKOR_invalid_v():
    with pytest.raises(ValueError):
        VIKOR(v=1.1)
    with pytest.raises(ValueError):
        VIKOR(v=-0.1)


@pytest.mark.parametrize("alt", [True, False])
def test_VIKOR_opricovic2004compromise(alt):
    """
    Data from:
        Opricovic, S., & Tzeng, G. H. (2004).
        Compromise solution by MCDM methods:
        A comparative analysis of VIKOR and TOPSIS.
        European Journal of Operational Research, 156(2), 445-455.

    """

    matrix = [
        [1.0, 3000.0],
        [2.0, 3750.0],
        [5.0, 4500.0],
    ]
    if alt:
        matrix = np.apply_along_axis(
            lambda row: (row[0] + 5, row[1] / 1000 - 1), 1, matrix
        )

    dm = skcriteria.mkdm(
        matrix=matrix,
        weights=[0.5, 0.5],
        objectives=[min, max],
        alternatives=["A1", "A2", "A3"],
        criteria=["Risk", "Altitude"],
    )

    expected = RankResult(
        "VIKOR",
        ["A1", "A2", "A3"],
        [2, 1, 2],
        {
            "r_k": np.array([0.5, 0.25, 0.5]),
            "s_k": np.array([0.5, 0.375, 0.5]),
            "q_k": np.array([1.0, 0.0, 1.0]),
            "acceptable_advantage": True,
            "acceptable_stability": True,
            "compromise_set": np.array([1]),
        },
    )

    ranker = VIKOR()
    result = ranker.evaluate(dm)
    assert expected.aequals(result)


@pytest.mark.parametrize("W", [1, 2, 3, 4])
def test_VIKOR_opricovic2007extended(W):
    """
    Data from:
        Opricovic, S., & Tzeng, G. H. (2007).
        Extended VIKOR method in comparison with outranking methods.
        European journal of operational research, 178(2), 514-529.
    """

    matrix = np.array(
        [
            [4184.3, 2914.0, 407.2, 251.0, 195, 244, 15, 2.41],
            [5211.9, 3630.0, 501.7, 308.3, 282, 346, 21, 1.41],
            [5021.3, 3920.5, 504.0, 278.6, 12, 56, 3, 4.42],
            [5566.1, 3957.9, 559.5, 335.3, 167, 268, 16, 3.36],
            [5060.5, 3293.5, 514.1, 284.2, 69, 90, 7, 4.04],
            [4317.9, 2925.9, 432.8, 239.3, 12, 55, 3, 4.36],
        ],
        dtype=float,
    )

    criteria = [
        "Profit [10e6 Din]",
        "Cost [10e6 Din]",
        "Energy produced [GW hour]",
        "Peak energy produced [GW hour]",
        "Homes to be relocated [Num]",
        "Reservoirs area [ha]",
        "Villages to displace [Num]",
        "Environmental protection [Grade]",
    ]

    alternatives = ["A1", "A2", "A3", "A4", "A5", "A6"]
    objectives = [max, min, max, max, min, min, min, max]

    if W == 1:
        weights = [1, 1, 1, 1, 1, 1, 1, 1]
        expected_r_k = [0.125, 0.125, 0.121, 0.125, 0.067, 0.125]
        expected_s_k = [0.692, 0.7, 0.29, 0.423, 0.28, 0.346]
        expected_q_k = [0.991, 1.0, 0.473, 0.670, 0.0, 0.578]
        expected_rank = [5, 6, 2, 4, 1, 3]
        expected_has_acceptable_advantage = True
        expected_has_acceptable_stability = True
        expected_compromise_set = np.array([4])
    elif W == 2:
        weights = [2, 2, 2, 2, 1, 1, 1, 1]
        expected_r_k = [0.167, 0.114, 0.161, 0.167, 0.089, 0.167]
        expected_s_k = [0.701, 0.6, 0.386, 0.365, 0.317, 0.459]
        expected_q_k = [1.0, 0.533, 0.552, 0.563, 0.0, 0.686]
        expected_rank = [6, 2, 3, 4, 1, 5]
        expected_has_acceptable_advantage = True
        expected_has_acceptable_stability = True
        expected_compromise_set = np.array([4])
    elif W == 3:
        weights = [1, 1, 1, 1, 2, 2, 2, 2]
        expected_r_k = [0.113, 0.167, 0.08, 0.122, 0.044, 0.083]
        expected_s_k = [0.683, 0.8, 0.193, 0.48, 0.243, 0.232]
        expected_q_k = [0.684, 1.0, 0.147, 0.554, 0.041, 0.191]
        expected_rank = [5, 6, 2, 4, 1, 3]
        expected_has_acceptable_advantage = False
        expected_has_acceptable_stability = True
        expected_compromise_set = np.array([2, 4, 5])
    elif W == 4:
        weights = [1, 1, 1, 1, 3.2, 3.2, 3.2, 3.2]
        expected_r_k = [0.129, 0.190, 0.057, 0.139, 0.042, 0.060]
        expected_s_k = [0.678, 0.857, 0.138, 0.513, 0.222, 0.167]
        expected_q_k = [0.668, 1.0, 0.051, 0.588, 0.058, 0.078]
        expected_rank = [5, 6, 1, 4, 2, 3]
        expected_has_acceptable_advantage = False
        expected_has_acceptable_stability = True
        expected_compromise_set = np.array([2, 4, 5])
    weights /= np.sum(weights)

    dm = skcriteria.mkdm(
        matrix=matrix,
        criteria=criteria,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
    )

    expected = RankResult(
        "VIKOR",
        ["A1", "A2", "A3", "A4", "A5", "A6"],
        expected_rank,
        {
            "r_k": np.array(expected_r_k),
            "s_k": np.array(expected_s_k),
            "q_k": np.array(expected_q_k),
            "acceptable_advantage": expected_has_acceptable_advantage,
            "acceptable_stability": expected_has_acceptable_stability,
            "compromise_set": np.array(expected_compromise_set),
        },
    )

    ranker = VIKOR(use_compromise_set=False)
    result = ranker.evaluate(dm)
    assert expected.aequals(result, rtol=1e-3, atol=1e-3, equal_nan=True)


@pytest.mark.parametrize("use_compromise_set", [True, False])
def test_VIKOR_acceptable_advantage_but_not_stability(use_compromise_set):
    matrix = np.array(
        [
            [1.0, 101.0, 1.0, 101.0, 1.00],
            [101.0, 1.0, 101.0, 1.0, 101.0],
            [1.0, 1.0, 1.0, 1.0, 101.0],
            [11.0, 11.0, 11.0, 16.0, 61.0],
            [56.0, 56.0, 56.0, 56.0, 56.0],
        ]
    )

    dm = skcriteria.mkdm(
        matrix=matrix,
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        objectives=[min, min, min, min, min],
        alternatives=["A1", "A2", "A3", "A4", "A5"],
        criteria=["C1", "C2", "C3", "C4", "C5"],
    )

    if use_compromise_set:
        expected_rank = [3, 4, 2, 1, 1]
    else:
        expected_rank = [4, 5, 3, 1, 2]
    expected = RankResult(
        "VIKOR",
        ["A1", "A2", "A3", "A4", "A5"],
        expected_rank,
        {
            "r_k": np.array([0.2, 0.2, 0.2, 0.12, 0.11]),
            "s_k": np.array([0.4, 0.6, 0.2, 0.21, 0.55]),
            "q_k": np.array([0.75, 1, 0.5, 0.068, 0.4375]),
            "acceptable_advantage": True,
            "acceptable_stability": False,
            "compromise_set": np.array([3, 4]),
        },
    )

    ranker = VIKOR(use_compromise_set=use_compromise_set)
    result = ranker.evaluate(dm)
    assert expected.aequals(result, atol=1e-3, rtol=1e-3)
