#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.edas"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.edas import EDAS


def test_EDAS_weights_sum_fail():
    """
    Same decision matrix as robot_example_2 with it's last weight modified.
    """
    dm = skcriteria.mkdm(
        matrix=[
            [60.0, 0.40, 2540, 500, 990],
            [6.35, 0.15, 1016, 3000, 1041],
            [6.80, 0.10, 1727.2, 1500, 1676],
            [10.0, 0.20, 1000, 2000, 965],
            [2.50, 0.10, 560, 500, 915],
            [4.50, 0.08, 1016, 350, 508],
            [3.00, 0.10, 177, 1000, 920],
        ],
        objectives=[max, min, max, max, max],
        weights=[0.1574, 0.1825, 0.2385, 0.2172, 0.2043],
        criteria=[
            "Load Capacity",
            "Repeatability",
            "Max Tip Speed",
            "Memory",
            "Reach",
        ],
        alternatives=["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
    )

    ranker = EDAS()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_EDAS_bad_weights_1_fail():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 16, 12, 5],
            [200, 16, 8, 3],
            [300, 32, 16, 4],
            [275, 32, 8, 4],
            [225, 16, 16, 2],
        ],
        objectives=[min, max, max, max],
        weights=[0.35, -0.2, 0.25, 0.15],
    )

    ranker = EDAS()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_EDAS_bad_weights_2_fail():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 16, 12, 5],
            [200, 16, 8, 3],
            [300, 32, 16, 4],
            [275, 32, 8, 4],
            [225, 16, 16, 2],
        ],
        objectives=[min, max, max, max],
        weights=[0.35, 1.2, 0.25, 0.15],
    )

    ranker = EDAS()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_EDAS_mobile_selection():
    """Source: https://www.youtube.com/watch?v=0ZHz4EeYB2Y."""
    dm = skcriteria.mkdm(
        matrix=[
            [250, 16, 12, 5],
            [200, 16, 8, 3],
            [300, 32, 16, 4],
            [275, 32, 8, 4],
            [225, 16, 16, 2],
        ],
        objectives=[min, max, max, max],
        weights=[0.35, 0.25, 0.25, 0.15],
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4"],
        [3, 5, 1, 2, 4],
        {"score": [0.4421, 0.169, 0.8053, 0.4697, 0.4015]},
    )

    assert result.values_equals(expected)
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-4)


def test_EDAS_notebook_selection():
    """Source: http://doi.org/10.17270/J.LOG.2021.603."""
    dm = skcriteria.mkdm(
        matrix=[
            [256, 8, 41, 1.6, 1.77, 7347.16],
            [256, 8, 32, 1.0, 1.8, 6919.99],
            [256, 8, 53, 1.6, 1.9, 8400],
            [256, 8, 41, 1.0, 1.75, 6808.9],
            [512, 8, 35, 1.6, 1.7, 8479.99],
            [256, 4, 35, 1.6, 1.7, 7499.99],
        ],
        objectives=[max, max, max, max, min, min],
        weights=[0.405, 0.221, 0.134, 0.199, 0.007, 0.034],
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4", "A5"],
        [3, 5, 2, 4, 1, 6],
        {"score": [0.414, 0.130, 0.461, 0.212, 0.944, 0.043]},
    )

    assert result.values_equals(expected)
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-1)


def test_EDAS_set_1():
    """Source: http://dx.doi.org/10.15388/Informatica.2015.57."""
    dm = skcriteria.mkdm(
        matrix=[
            [23, 264, 2.37, 0.05, 167, 8900, 8.71],  # A1
            [20, 220, 2.20, 0.04, 171, 9100, 8.23],  # A2
            [17, 231, 1.98, 0.15, 192, 10800, 9.91],  # A3
            [12, 210, 1.73, 0.20, 195, 12300, 10.21],  # A4
            [15, 243, 2.00, 0.14, 187, 12600, 9.34],  # A5
            [14, 222, 1.89, 0.13, 180, 13200, 9.22],  # A6
            [21, 262, 2.43, 0.06, 160, 10300, 8.93],  # A7
            [20, 256, 2.60, 0.07, 163, 11400, 8.44],  # A8
            [19, 266, 2.10, 0.06, 157, 11200, 9.04],  # A9
            [8, 218, 1.94, 0.11, 190, 13400, 10.11],  # A10
        ],
        objectives=[max, max, max, min, min, min, min],
        weights=[0.250, 0.214, 0.179, 0.143, 0.107, 0.071, 0.036],
        criteria=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
        [1, 4, 6, 10, 7, 8, 2, 3, 5, 9],
        {"score": []},
    )

    assert result.values_equals(expected)


def test_EDAS_electric_motorcycles():
    """Source: https://doi.org/10.46632/jame/2/4/2."""
    dm = skcriteria.mkdm(
        matrix=[
            [3.20, 150, 80, 129.400, 4.5],  # Revolt RV400
            [2.80, 75, 25, 102.249, 4.5],  # Joy e-Bike Monster
            [4.00, 180, 105, 192.499, 5.0],  # Tork Kratos R
            [3.60, 200, 80, 114.999, 6.0],  # Komaki Ranger
            [2.88, 110, 85, 114.999, 5.0],  # Cyborg Bob
            [4.32, 140, 80, 166.250, 6.0],  # Odysse Evogis
            [4.40, 200, 100, 99.999, 2.0],  # Oben Rorr
            [3.50, 140, 85, 154.999, 6.0],  # PURE EV eTryst 350
            [3.00, 135, 75, 99.999, 3.0],  # Pure ecoDryft
        ],
        objectives=[max, max, max, min, min],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        criteria=[
            "Battery (kWh)",
            "Range (km)",
            "Top Speed (kmph)",
            "Price (k₹)",
            "Charging Time (hrs)",
        ],
        alternatives=[
            "Revolt RV400",
            "Joy e-Bike Monster",
            "Tork Kratos R",
            "Komaki Ranger",
            "Cyborg Bob",
            "Odysse Evogis",
            "Oben Rorr",
            "PURE EV eTryst 350",
            "Pure ecoDryft",
        ],
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "EDAS",
        [
            "Revolt RV400",
            "Joy e-Bike Monster",
            "Tork Kratos R",
            "Komaki Ranger",
            "Cyborg Bob",
            "Odysse Evogis",
            "Oben Rorr",
            "PURE EV eTryst 350",
            "Pure ecoDryft",
        ],
        [5, 9, 4, 3, 6, 7, 1, 8, 2],
        {
            "score": [
                0.48700,
                0.07582,
                0.50536,
                0.54723,
                0.37261,
                0.34899,
                1.00000,
                0.32880,
                0.57229,
            ]
        },
    )

    assert result.values_equals(expected)
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-1)
