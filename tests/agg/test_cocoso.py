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
from skcriteria.preprocessing.scalers import VectorScaler


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_COCOSO():
    """
    Data From:
        Yazdani, Morteza and Zaraté, Pascale and Kazimieras Zavadskas,
        Edmundas and Turskis, Zenonas
        A Combined Compromise Solution (CoCoSo) method for multi-criteria decision-making problems.
        (2019) Management Decision, 57 (9). 2501-2519. ISSN 0025-1747
    """

    dm = skcriteria.mkdm(
        matrix=[
            [60,    0.4,    2_540,      500,    990],
            [6.35,  0.15,   1_016,      3_000,  1_041],
            [6.8,   0.1,    1_727.2,    1_500,  1_676],
            [10,    0.2,    1_000,      2_000,  965],
            [2.5,   0.1,    560,        500,    915],
            [4.5,   0.08,   1_016,      350,    508],
            [3,     0.1,    1_778,      1_000,  920]
        ],
        objectives=[max, min, max, max, max],
        weights=[0.036, 0.192, 0.326, 0.326, 0.12],
    )

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
            ]
        },
    )

    transformer = VectorScaler(target="matrix")
    dm = transformer.transform(dm)

    ranker = CoCoSo()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-4)

