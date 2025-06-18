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

    diff = expected.diff(result)
    assert not diff.has_differences, diff


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

    assert expected == result


def test_VIKOR_opricovic2007extended():
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
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    weights /= np.sum(weights)

    dm = skcriteria.mkdm(
        matrix=matrix,
        criteria=criteria,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
    )

    q_k = np.array([0.991, 1.0, 0.473, 0.670, 0.0, 0.578])
    s_k = np.array([0.692, 0.7, 0.29, 0.423, 0.28, 0.346])
    r_k = np.array([0.125, 0.125, 0.121, 0.125, 0.067, 0.125])

    expected = RankResult(
        "VIKOR",
        ["A1", "A2", "A3", "A4", "A5", "A6"],
        [5, 6, 2, 4, 1, 3],
        {
            "r_k": r_k,
            "s_k": s_k,
            "q_k": q_k,
            "acceptable_advantage": True,
            "acceptable_stability": True,
            "compromise_set": np.array([5]),
        },
    )

    ranker = VIKOR()
    result = ranker.evaluate(dm)

    assert expected.aequals(result, rtol=1e-3, atol=1e-3, equal_nan=True)
