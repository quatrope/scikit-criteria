#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.edas."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.edas import EDAS


# =============================================================================
# TESTS
# =============================================================================


def test_edas_mobile_selection():
    """
    Data From:
        Manoj Mathew. (2018, July 17).
        Evaluation Based on Distance from Average Solution - EDAS.
        https://www.youtube.com/watch?v=0ZHz4EeYB2Y
    """
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

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4"],
        [3, 5, 1, 2, 4],
        {"score": [0.4421, 0.169, 0.8053, 0.4697, 0.4015]},
    )

    ranker = EDAS()

    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_edas_notebook_selection():
    """
    Data From:
        Ersoy, Y. (2021).
        Equipment selection for an e-commerce company using Entropy-based
        TOPSIS, EDAS and CODAS methods during the COVID-19.
        LogForum, 17(3).
    """
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

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4", "A5"],
        [3, 5, 2, 4, 1, 6],
        {"score": [0.414, 0.130, 0.461, 0.212, 0.944, 0.043]},
    )

    ranker = EDAS()

    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_edas_electric_motorcycles():
    """
    Data From:
        Sharma, R., Ramachandran, M., Saravanan, V., & Nanjundan, P.
        Application of the EDAS Technique for Selecting the Electric
        Motor Vehicles.
    """
    dm = skcriteria.mkdm(
        matrix=[
            [3.20, 150, 80, 129.400, 4.5],
            [2.80, 75, 25, 102.249, 4.5],
            [4.00, 180, 105, 192.499, 5.0],
            [3.60, 200, 80, 114.999, 6.0],
            [2.88, 110, 85, 114.999, 5.0],
            [4.32, 140, 80, 166.250, 6.0],
            [4.40, 200, 100, 99.999, 2.0],
            [3.50, 140, 85, 154.999, 6.0],
            [3.00, 135, 75, 99.999, 3.0],
        ],
        objectives=[max, max, max, min, min],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    )

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
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

    ranker = EDAS()

    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-3)


def test_edas_tech_company():
    """
    Data From:
        Karabasevic, D., Zavadskas, E. K., Stanujkic, D., Popovic, G.,
        & Brzakovic, M. (2018).
        An approach to personnel selection in the IT industry based
        on the EDAS method.
        In Transformations in business & economics
        (Vol. 17, No. 2 (44), pp. 54-65).
    """
    dm = skcriteria.mkdm(
        matrix=[
            [5, 4, 3, 4, 4, 5, 3],
            [3, 4, 5, 4, 3, 3, 4],
            [4, 3, 2, 3, 2, 3, 4],
            [3, 3, 3, 4, 4, 3, 4],
            [4, 3, 3, 4, 4, 4, 3],
            [5, 4, 4, 5, 5, 5, 4],
        ],
        objectives=[max, max, max, max, max, max, max],
        weights=[0.31, 0.21, 0.17, 0.13, 0.09, 0.06, 0.03],
    )

    expected = RankResult(
        "EDAS",
        ["A0", "A1", "A2", "A3", "A4", "A5"],
        [2, 3, 6, 5, 4, 1],
        {"score": [0.73, 0.47, 0.01, 0.16, 0.38, 1.00]},
    )

    ranker = EDAS()

    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1.0e-2)


def test_edas_zero_in_matrix():
    dm = skcriteria.mkdm(
        matrix=[
            [1, 0, 3],
            [2, 0, 4],
            [3, 0, 5],
        ],
        objectives=[max, max, max],
        weights=[0.3, 0.4, 0.3],
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    # Assert that the evaluation completes without errors
    # and produces a RankResult
    assert isinstance(result, RankResult)
    assert len(result.alternatives) == 3
