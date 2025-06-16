#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.aras."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.aras import ARAS
from skcriteria.preprocessing import invert_objectives
from skcriteria.preprocessing.scalers import SumScaler

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_ARAS_balezentiene2012reducing():
    """
    Data from:
        Balezentiene, L., & Kusta, A. (2012).
        Reducing Greenhouse Gas Emissions in Grassland Ecosystems of the
        Central Lithuania: Multi-Criteria Evaluation on a Basis of the ARAS
        Method.

    """
    dm = skcriteria.mkdm(
        [
            [
                3020.0,
                827.6948,
                98,
                0.015479,
                2.166783,
                0.0141,
            ],  # Ideal solution
            [957.5, 190.1977, 70, 0.039592, 2.166783, 0.0141],  # Control
            [892.5, 203.8013, 53, 0.025849, 2.994347, 0.016642],  # N_60
            [1002.5, 235.0942, 75, 0.015479, 4.742146, 0.019484],  # N_120
            [1150.0, 271.4, 76, 0.022, 7.055962, 0.022147],  # N_180
            [1520.0, 192.5579, 70, 0.025442, 8.424319, 0.024235],  # N_240
            [
                1700.0,
                386.1619,
                90,
                0.029429,
                5.888983,
                0.022218,
            ],  # N_180 P_120
            [1355.0, 342.1966, 80, 0.0293, 8.635847, 0.024861],  # N_180 K_150
            [
                2127.5,
                495.4876,
                90,
                0.038453,
                3.652983,
                0.020537,
            ],  # N_60 P_40 K_50
            [
                3020.0,
                827.6948,
                97,
                0.031774,
                11.03944,
                0.024359,
            ],  # N_180 P_120 K_150
            [
                2786.0,
                795.0,
                98,
                0.021151,
                8.952235,
                0.023349,
            ],  # CP(N_180 P_120 K_150)
        ],
        objectives=[max, max, max, min, min, min],
        weights=([0.166667] * 6),
    )

    inverter = invert_objectives.InvertMinimize()
    dm = inverter.transform(dm)

    scaler = SumScaler(target="matrix")
    dm = scaler.transform(dm)

    expected = RankResult(
        "ARAS",
        ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
        [4, 7, 5, 8, 10, 6, 9, 3, 2, 1],
        {
            "score": [
                0.08926,
                0.079211,
                0.084055,
                0.073215,
                0.067523,
                0.082115,
                0.070786,
                0.094696,
                0.102704,
                0.107538,
            ],
            "utility": [
                0.599472,
                0.531983,
                0.564514,
                0.491713,
                0.453489,
                0.551491,
                0.475401,
                0.635985,
                0.689763,
                0.722232,
            ],
            "ideal_score": 0.148897,
        },
    )

    ranker = ARAS()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-4)
    assert np.allclose(result.e_.utility, expected.e_.utility, atol=1.0e-4)
    assert np.allclose(
        result.e_.ideal_score, expected.e_.ideal_score, atol=1.0e-4
    )


def test_ARAS_invalid_min_objective():
    dm = skcriteria.mkdm(
        matrix=[[1, 5, 6], [1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = ARAS()
    with pytest.raises(
        ValueError, match="ARAS can't operate with minimization objectives"
    ):
        ranker.evaluate(dm)


def test_ARAS_bad_ideal_valueError():
    dm = skcriteria.mkdm(
        matrix=[[1, 3, 2], [1, 0, 3], [0, 5, 6]],  # ideal
        objectives=[max, max, max],
    )

    ranker = ARAS()
    with pytest.raises(ValueError, match="Invalid ideal vector"):
        ranker.evaluate(dm)
