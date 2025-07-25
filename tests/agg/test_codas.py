#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.codas."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.codas import CODAS
from skcriteria.preprocessing.invert_objectives import BenefitCostInverter


# =============================================================================
# TESTS
# =============================================================================


def test_CODAS_incorrect_dm_1():
    dm = skcriteria.mkdm(
        matrix=[[0.1, 1.2, 0.3], [0.4, -1, 0.6]],
        objectives=[max, max, max],
        weights=[0.5, 0.3, 0.2],
    )

    ranker = CODAS()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_CODAS_incorrect_dm_2():
    dm = skcriteria.mkdm(
        matrix=[[0.1, 0.2, 0.3], [0.4, 0.9, 0.6]],
        objectives=[max, min, max],
        weights=[0.5, 0.3, 0.2],
    )

    ranker = CODAS()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_CODAS_lisco():
    """
    Data From:
        Badi, I., Shetwan, A. G., & Abdulshahed, A. M. (2017, September).
        Supplier selection using COmbinative Distance-based ASsessment (CODAS)
        method for multi-criteria decision-making.
        In Proceedings of the 1st international conference on management,
        engineering and environment (ICMNEE) (pp. 395-407).
    """
    dm = skcriteria.mkdm(
        matrix=[
            [45, 3600, 45, 0.9],
            [25, 3800, 60, 0.8],
            [23, 3100, 35, 0.9],
            [14, 3400, 50, 0.7],
            [15, 3300, 40, 0.8],
            [28, 3000, 30, 0.6],
        ],
        objectives=[max, min, max, max],
        weights=[0.2857, 0.3036, 0.2321, 0.1786],
    )

    expected = RankResult(
        "CODAS",
        ["A0", "A1", "A2", "A3", "A4", "A5"],
        [1, 2, 3, 5, 6, 4],
        {"score": [1.3914, 0.3411, -0.2170, -0.5381, -0.7292, -0.2481]},
    )

    transformer = BenefitCostInverter()
    ranker = CODAS()

    dm_transformed = transformer.transform(dm)
    result = ranker.evaluate(dm_transformed)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_CODAS_libya():
    """
    Data From:
        Badi, I., Ballem, M., & Shetwan, A. (2018).
        SITE SELECTION OF DESALINATION PLANT IN LIBYA BY USING
        COMBINATIVE DISTANCE-BASED ASSESSMENT (CODAS) METHOD.
        International Journal for Quality Research, 12(3).
    """
    dm = skcriteria.mkdm(
        matrix=[
            [8, 8, 10, 9, 5],
            [8, 9, 9, 9, 8],
            [9, 9, 7, 8, 6],
            [8, 8, 7, 8, 9],
            [9, 8, 7, 7, 4],
        ],
        objectives=[min, max, max, max, min],
        weights=[0.19, 0.26, 0.24, 0.17, 0.14],
    )

    expected = RankResult(
        "CODAS",
        ["A0", "A1", "A2", "A3", "A4"],
        [1, 2, 4, 5, 3],
        {"score": [0.4463, 0.1658, -0.2544, -0.4618, 0.1041]},
    )

    transformer = BenefitCostInverter()
    dm_transformed = transformer.transform(dm)

    ranker = CODAS()
    result = ranker.evaluate(dm_transformed)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_CODAS_bakir_atalik_2018():
    """
    Data From:
        BAKIR, M., & ALPTEKİN, N. (2018).
        A new approach in service quality assessment: An application on
        airlines through CODAS method.
        Business & Management Studies: An International Journal, 6(4), 1336.
    """
    dm = skcriteria.mkdm(
        matrix=[
            [3.100, 2.714, 2.750, 3.500, 3.167, 2.700, 3.219],
            [3.750, 3.929, 4.000, 3.938, 3.667, 3.750, 3.563],
            [4.750, 4.125, 4.000, 4.800, 4.500, 4.625, 4.800],
            [3.500, 3.250, 3.750, 3.300, 3.000, 3.375, 4.200],
            [3.900, 4.071, 4.125, 4.000, 3.667, 3.000, 3.907],
            [3.500, 4.071, 3.750, 3.688, 3.667, 4.417, 3.625],
            [4.250, 3.571, 3.875, 4.875, 4.500, 3.875, 4.187],
            [2.800, 3.000, 3.000, 3.250, 2.667, 3.500, 3.000],
            [3.750, 4.143, 3.375, 4.000, 3.667, 3.800, 3.687],
            [3.900, 4.214, 4.000, 4.125, 4.000, 4.125, 3.969],
            [3.500, 4.429, 3.750, 3.875, 4.000, 4.100, 3.844],
        ],
        objectives=[max, max, max, max, max, max, max],
        weights=[0.1468, 0.1661, 0.1116, 0.1287, 0.1799, 0.1466, 0.1203],
    )

    expected = RankResult(
        "CODAS",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
        [11, 7, 1, 9, 6, 5, 2, 10, 8, 3, 4],
        {
            "score": [
                -2.4717,
                0.1479,
                2.4227,
                -1.0873,
                0.2287,
                0.325,
                1.2208,
                -2.4066,
                0.1457,
                0.8532,
                0.6216,
            ]
        },
    )

    transformer = BenefitCostInverter()
    ranker = CODAS()

    dm_transformed = transformer.transform(dm)
    result = ranker.evaluate(dm_transformed)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_CODAS_cloud_service():
    """
    Data From:
        Baki, R. (2022).
        Application of ROC and CODAS techniques for cloud service
        provider selection.
        Gaziantep University Journal of Social Sciences, 21(1), 217-230.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [3.464, 2.942, 2.667, 2.936, 3.595, 3.026, 3.659, 3.957],
            [2.749, 3.634, 2.182, 2.804, 2.994, 3.360, 2.182, 3.464],
            [4.263, 3.175, 4.107, 2.621, 2.942, 4.472, 3.772, 3.360],
            [2.621, 2.289, 2.289, 3.634, 2.621, 3.086, 2.289, 2.749],
        ],
        objectives=[max, max, max, max, max, max, max, max],
        weights=[0.11, 0.092, 0.217, 0.02105, 0.198, 0.037, 0.267, 0.058],
    )

    expected = RankResult(
        "CODAS",
        ["A0", "A1", "A2", "A3"],
        [2, 3, 1, 4],
        {"score": [0.476, -0.536, 0.924, -0.864]},
    )

    transformer = BenefitCostInverter()
    ranker = CODAS()

    dm_transformed = transformer.transform(dm)
    result = ranker.evaluate(dm_transformed)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-2)


def test_CODAS_zavadaskas_turskis_2010():
    """
    Data From:
        TURSKIS, Z., & ANTUCHEVICIENE, J. (2016).
        A NEW COMBINATIVE DISTANCE-BASED ASSESSMENT (CODAS)
        METHOD FOR MULTI-CRITERIA DECISION-MAKING.
    """
    dm = skcriteria.mkdm(
        matrix=[
            [7.6, 46, 18, 390, 0.1, 11],
            [5.5, 32, 21, 360, 0.05, 11],
            [5.3, 32, 21, 290, 0.05, 11],
            [5.7, 37, 19, 270, 0.05, 9],
            [4.2, 38, 19, 240, 0.1, 8],
            [4.4, 38, 19, 260, 0.1, 8],
            [3.9, 42, 16, 270, 0.1, 5],
            [7.9, 44, 20, 400, 0.05, 6],
            [8.1, 44, 20, 380, 0.05, 6],
            [4.5, 46, 18, 320, 0.1, 7],
            [5.7, 48, 20, 320, 0.05, 11],
            [5.2, 48, 20, 310, 0.05, 11],
            [7.1, 49, 19, 280, 0.1, 12],
            [6.9, 50, 16, 250, 0.05, 10],
        ],
        objectives=[max, max, max, max, min, min],
        weights=[0.21, 0.16, 0.26, 0.17, 0.12, 0.08],
    )

    expected = RankResult(
        "CODAS",
        [
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
        ],
        [
            3,
            6,
            9,
            10,
            14,
            13,
            12,
            1,
            2,
            11,
            4,
            7,
            8,
            5,
        ],
        {
            "score": [
                0.768,
                0.363,
                -0.105,
                -0.329,
                -2.384,
                -2.207,
                -2.043,
                2.929,
                2.890,
                -1.282,
                0.568,
                0.313,
                0.157,
                0.364,
            ]
        },
    )

    transformer = BenefitCostInverter()
    ranker = CODAS()

    dm_transformed = transformer.transform(dm)
    result = ranker.evaluate(dm_transformed)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_CODAS_tau_warning():
    dm = skcriteria.mkdm(
        matrix=[[0.1, 0.2, 0.3], [0.4, 0.9, 0.6]],
        objectives=[max, max, max],
        weights=[0.5, 0.3, 0.2],
    )

    # Test with tau < 0.01
    ranker_low_tau = CODAS(tau=0.005)
    with pytest.warns(
        UserWarning,
        match="It is suggested to set tau at a value between 0.01 and 0.05",
    ):
        ranker_low_tau.evaluate(dm)

    # Test with tau > 0.05
    ranker_high_tau = CODAS(tau=0.06)
    with pytest.warns(
        UserWarning,
        match="It is suggested to set tau at a value between 0.01 and 0.05",
    ):
        ranker_high_tau.evaluate(dm)
