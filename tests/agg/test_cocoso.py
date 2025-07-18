#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.cocoso."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.cocoso import CoCoSo


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_COCOSO():
    """
    Data From:
        Yazdani, Morteza and Zaraté, Pascale and Kazimieras Zavadskas,
        Edmundas and Turskis, Zenonas
        A Combined Compromise Solution (CoCoSo) method for multi-criteria
        decision-making problems.
        (2019) Management Decision, 57 (9). 2501-2519. ISSN 0025-1747
    """

    dm = skcriteria.mkdm(
        matrix=[
            [1, 0, 1, 0.0566, 0.4127],
            [0.067, 0.7813, 0.2303, 1, 0.4563],
            [0.0748, 0.9375, 0.5895, 0.434, 1],
            [0.1304, 0.625, 0.2222, 0.6226, 0.3913],
            [0, 0.9375, 0, 0.0566, 0.3485],
            [0.0348, 1, 0.2303, 0, 0],
            [0.0087, 0.9375, 0.6152, 0.2453, 0.3527],
        ],
        objectives=[max, min, max, max, max],
        weights=[0.036, 0.192, 0.326, 0.326, 0.12],
    )

    lambda_value = 0.5

    expected = RankResult(
        "CoCoso",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
        [5, 2, 1, 4, 7, 6, 3],
        {
            "score": [
                2.041,
                2.788,
                2.882,
                2.416,
                1.3,
                1.443,
                2.52,
            ],
        },
    )

    ranker = CoCoSo(lambda_value)
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-2)


def test_COCOSO_invalid_lambda_value():
    with pytest.raises(ValueError):
        CoCoSo(lambda_value=1.5)

    with pytest.raises(ValueError):
        CoCoSo(lambda_value=-1)


def test_COCOSO_negative_values_fail():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, -1, 6]],
        objectives=[max, max, max],
        weights=[0.036, 0.192, 0.326],
    )

    ranker = CoCoSo()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)
