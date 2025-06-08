#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.maut"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.simple import (
    WeightedAggregatedSumProductAssessment,
    SPOTIS,
    WeightedProductModel,
    WeightedSumModel,
)
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_WeightedSumModel():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, max, max],
    )

    expected = RankResult(
        "WeightedSumModel", ["A0", "A1"], [2, 1], {"score": [4.0, 11.0]}
    )

    ranker = WeightedSumModel()

    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.score == expected.e_.score)


def test_WeightedSumModel_minimize_fail():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = WeightedSumModel()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_WeightedProductModel_lt0_fail():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, -1, 6]],
        objectives=[max, max, max],
    )

    ranker = WeightedSumModel()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_WeightedSumModel_kracka2010ranking():
    """
    Data from:
        KRACKA, M; BRAUERS, W. K. M.; ZAVADSKAS, E. K. Ranking
        Heating Losses in a Building by Applying the MULTIMOORA. -
        ISSN 1392 - 2785 Inzinerine Ekonomika-Engineering Economics, 2010,
        21(4), 352-359.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [33.95, 23.78, 11.45, 39.97, 29.44, 167.10, 3.852],
            [38.9, 4.17, 6.32, 0.01, 4.29, 132.52, 25.184],
            [37.59, 9.36, 8.23, 4.35, 10.22, 136.71, 10.845],
            [30.44, 37.59, 13.91, 74.08, 45.10, 198.34, 2.186],
            [36.21, 14.79, 9.17, 17.77, 17.06, 148.3, 6.610],
            [37.8, 8.55, 7.97, 2.35, 9.25, 134.83, 11.935],
        ],
        objectives=[min, min, min, min, max, min, max],
        weights=[20, 20, 20, 20, 20, 20, 20],
    )

    transformers = [
        InvertMinimize(),
        SumScaler(target="both"),
    ]
    for t in transformers:
        dm = t.transform(dm)

    expected = RankResult(
        "WeightedSumModel",
        ["A0", "A1", "A2", "A3", "A4", "A5"],
        [6, 1, 3, 4, 5, 2],
        {
            "score": [
                0.12040426,
                0.3458235,
                0.13838192,
                0.12841246,
                0.12346084,
                0.14351701,
            ]
        },
    )

    ranker = WeightedSumModel()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score)


# =============================================================================
# WPM
# =============================================================================


def test_WeightedProductModel():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[max, max, max],
    )

    expected = RankResult(
        "WeightedProductModel",
        ["A0", "A1"],
        [2, 1],
        {"score": [0.77815125, 2.07918125]},
    )

    ranker = WeightedProductModel()

    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score)


def test_WeightedProductModel_minimize_fail():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = WeightedProductModel()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_WeightedProductModel_with0_fail():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 0, 6]],
        objectives=[max, max, max],
    )

    ranker = WeightedProductModel()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_WeightedProductModel_enwiki_1015567716():
    """
    Data from:

        Weighted product model. (n.d.). Retrieved January 07, 2017,
        from http://en.wikipedia.org/wiki/Weighted_product_model

    """
    dm = skcriteria.mkdm(
        matrix=[
            [25, 20, 15, 30],
            [10, 30, 20, 30],
            [30, 10, 30, 10],
        ],
        objectives=[max, max, max, max],
        weights=[20, 15, 40, 25],
    )

    expected = RankResult(
        "WeightedProductModel",
        ["A0", "A1", "A2"],
        [1, 2, 3],
        {"score": [-0.50128589, -0.50448471, -0.52947246]},
    )

    transformer = SumScaler(target="both")
    dm = transformer.transform(dm)

    ranker = WeightedProductModel()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score)


# =============================================================================
# WASPAS
# =============================================================================


@pytest.mark.parametrize(
    "ranker_cls, lambda_value",
    [
        (WeightedSumModel, 1),
        (WeightedProductModel, 0),
    ],
)
def test_WASPAS_compared_to_known_models(ranker_cls, lambda_value):
    dm = skcriteria.mkdm(
        matrix=[
            [0.5099, 0.0465, 0.2427, 0.3246, 0.1166],
            [0.2123, 0.7622, 0.2051, 0.3259, 0.5295],
            [0.0280, 0.0456, 0.2134, 0.1151, 0.1359],
            [0.2496, 0.1455, 0.3386, 0.2342, 0.2177],
        ],
        objectives=[max, max, max, max, max],
        weights=[0.3672, 0.0933, 0.2405, 0.2770, 0.0217],
    )

    expected_ranker = ranker_cls()
    expected_result = expected_ranker.evaluate(dm)

    waspas_ranker = WeightedAggregatedSumProductAssessment(
        lambda_value=lambda_value
    )
    waspas_result = waspas_ranker.evaluate(dm)

    assert waspas_result.values_equals(expected_result)
    assert np.allclose(
        waspas_result.e_.score,
        (
            10**expected_result.e_.score
            if lambda_value == 0
            else expected_result.e_.score
        ),
        atol=1e-4,
    )


@pytest.mark.parametrize("lambda_value", [0, 0.25, 0.5, 0.75, 1])
def test_WASPAS_verify_intermediate_calculations(lambda_value):
    dm = skcriteria.mkdm(
        matrix=[
            [0.5099, 0.0465, 0.2427, 0.3246, 0.1166],
            [0.2123, 0.7622, 0.2051, 0.3259, 0.5295],
            [0.0280, 0.0456, 0.2134, 0.1151, 0.1359],
            [0.2496, 0.1455, 0.3386, 0.2342, 0.2177],
        ],
        objectives=[max, max, max, max, max],
        weights=[0.3672, 0.0933, 0.2405, 0.2770, 0.0217],
    )

    wsm_result = WeightedSumModel().evaluate(dm)
    wpm_result = WeightedProductModel().evaluate(dm)

    waspas_ranker = WeightedAggregatedSumProductAssessment(
        lambda_value=lambda_value
    )
    waspas_result = waspas_ranker.evaluate(dm)

    assert np.allclose(waspas_result.e_.wsm_scores, wsm_result.e_.score)
    assert np.allclose(waspas_result.e_.log10_wpm_scores, wpm_result.e_.score)


def test_WASPAS_with_minimize_fails():
    """
    WASPAS should raise ValueError if input
    matrix contains min objectives.
    """
    dm = skcriteria.mkdm(
        matrix=[[1, 7, 3], [3, 5, 6]],
        objectives=[max, min, max],
    )
    ranker = WeightedAggregatedSumProductAssessment()

    with pytest.raises(
        ValueError,
        match=(
            "WeightedAggregatedSumProductAssessment can't "
            "operate with minimize objective"
        ),
    ):
        ranker.evaluate(dm)


def test_WASPAS_with_zero_fails():
    """
    WASPAS should raise ValueError if matrix
     contains 0s (division/log problems).
    """
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 0, 6]],
        objectives=[max, max, max],
    )
    ranker = WeightedAggregatedSumProductAssessment()

    with pytest.raises(
        ValueError,
        match=(
            "WeightedAggregatedSumProductAssessment can't "
            "operate with values <= 0"
        ),
    ):
        ranker.evaluate(dm)


@pytest.mark.parametrize("invalid_lambda", [-0.1, 1.1, 2, -5])
def test_WASPAS_invalid_l_values(invalid_lambda):
    """WASPAS should raise ValueError if lambda_value is not in [0, 1]"""
    dm = skcriteria.mkdm(
        matrix=[[1, 2], [3, 4]],
        objectives=[max, max],
    )

    with pytest.raises(
        ValueError,
        match=(
            "WeightedAggregatedSumProductAssessment requires 'lambda_value'"
            f" to be between 0 and 1, but found {invalid_lambda}."
        ),
    ):
        ranker = WeightedAggregatedSumProductAssessment(
            lambda_value=invalid_lambda
        )
        ranker.evaluate(dm)


@pytest.mark.parametrize(
    "lambda_value, expected_rank, expected_scores",
    [
        (
            0,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8578,
                0.8144,
                0.8408,
                0.9413,
                0.8632,
                0.8241,
                0.8454,
                0.8915,
                0.8987,
                0.8697,
            ],
        ),
        (
            0.1,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8580,
                0.8149,
                0.8410,
                0.9414,
                0.8633,
                0.8249,
                0.8464,
                0.8922,
                0.8995,
                0.8706,
            ],
        ),
        (
            0.2,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8581,
                0.8154,
                0.8412,
                0.9415,
                0.8634,
                0.8256,
                0.8474,
                0.8929,
                0.9003,
                0.8714,
            ],
        ),
        (
            0.3,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8582,
                0.8159,
                0.8414,
                0.9415,
                0.8636,
                0.8263,
                0.8484,
                0.8936,
                0.9011,
                0.8723,
            ],
        ),
        (
            0.4,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8583,
                0.8164,
                0.8417,
                0.9416,
                0.8637,
                0.8271,
                0.8493,
                0.8942,
                0.9019,
                0.8732,
            ],
        ),
        (
            0.5,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8584,
                0.8169,
                0.8419,
                0.9417,
                0.8638,
                0.8278,
                0.8503,
                0.8949,
                0.9027,
                0.8740,
            ],
        ),
        (
            0.6,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8585,
                0.8174,
                0.8421,
                0.9417,
                0.8640,
                0.8286,
                0.8513,
                0.8956,
                0.9035,
                0.8749,
            ],
        ),
        (
            0.7,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8586,
                0.8179,
                0.8423,
                0.9418,
                0.8641,
                0.8293,
                0.8523,
                0.8963,
                0.9043,
                0.8758,
            ],
        ),
        (
            0.8,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8588,
                0.8184,
                0.8425,
                0.9419,
                0.8643,
                0.8300,
                0.8532,
                0.8969,
                0.9051,
                0.8766,
            ],
        ),
        (
            0.9,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8589,
                0.8189,
                0.8427,
                0.9419,
                0.8644,
                0.8308,
                0.8542,
                0.8976,
                0.9059,
                0.8775,
            ],
        ),
        (
            1,
            [6, 10, 8, 1, 5, 9, 7, 3, 2, 4],
            [
                0.8590,
                0.8194,
                0.8430,
                0.9420,
                0.8645,
                0.8315,
                0.8552,
                0.8983,
                0.9067,
                0.8784,
            ],
        ),
    ],
)
def test_WASPAS_chakraborty2015applications(
    lambda_value, expected_rank, expected_scores
):
    """
    Data from:

        Chakraborty, S., Zavadskas, E. K., & Antucheviciene, J. (2015).
        Applications of WASPAS method as a multi-criteria decision-making tool.
        Economic Computation and Economic Cybernetics Studies and Research,
        49(1), 5-22. Example 2.

        This data corresponds to Example 2 from the paper.
    """

    matrix = [
        [581818, 54.49, 3, 5500],
        [595454, 49.73, 3, 4500],
        [586060, 51.24, 3, 5000],
        [522727, 45.71, 3, 5800],
        [561818, 52.66, 3, 5200],
        [543030, 74.46, 4, 5600],
        [522727, 75.42, 4, 5800],
        [486970, 62.62, 4, 5600],
        [509394, 65.87, 4, 6400],
        [513333, 70.67, 4, 6000],
    ]

    # Normalize the matrix
    objectives = [min, min, min, max]
    norm_matrix = np.asarray(matrix, dtype=np.float64)
    for i, obj in enumerate(objectives):
        if obj is min:
            norm_matrix[:, i] = np.min(norm_matrix[:, i]) / norm_matrix[:, i]
        else:
            norm_matrix[:, i] = norm_matrix[:, i] / np.max(norm_matrix[:, i])

    dm = skcriteria.mkdm(
        matrix=norm_matrix,
        objectives=[max] * 4,
        weights=[0.467, 0.160, 0.095, 0.278],
        criteria=["PC", "FS", "MN", "P"],
    )

    expected = RankResult(
        "WeightedAggregatedSumProductAssessment",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
        expected_rank,
        {"score": expected_scores},
    )

    ranker = WeightedAggregatedSumProductAssessment(lambda_value=lambda_value)
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-4)

# =============================================================================
# SPOTIS
# =============================================================================

class TestSPOTISMethod:
    """Test suite for the SPOTIS method.
        
    Many tests are based on the reference examples from the method paper:

    Dezert, J., Tchamova, A., Han, D., & Tacnet, J. M. (2020, July). The SPOTIS rank reversal free method for multi-criteria decision-making support. In 2020 IEEE 23rd International Conference on Information Fusion (FUSION) (pp. 1-8). IEEE.
    """

    def test_reference_example_a(self):
        dm = skcriteria.mkdm(
            matrix=[
                [10.5, -3.1, 1.7],
                [-4.7, 0, 3.4],
                [8.1, 0.3, 1.3],
                [3.2, 7.3, -5.3]
            ],
            objectives=[max, min, max],
            weights=[0.2, 0.3, 0.5],
            criteria=["C1", "C2", "C3"],
        )
        bounds = np.array([
            [-5, 12],
            [-6, 10],
            [-8, 5]
        ])
        isp = np.array([12, -6, 5])

        result = SPOTIS(bounds=bounds, isp=isp).evaluate(dm)
        expected = RankResult(
            "SPOTIS",
            ["A0", "A1", "A2", "A3"],
            [1, 3, 2, 4],
            {"score": [0.1989, 0.3707, 0.3063, 0.7491], "isp": isp, "bounds": bounds}
        )

        self.assert_rank_result(result, expected, atol=1e-3)

    def test_reference_example_b(self):
        dm = skcriteria.mkdm(
            matrix=[
                [15000, 4.3, 99, 42, 737],
                [15290, 5.0, 116, 42, 892],
                [15350, 5.0, 114, 45, 952],
                [15490, 5.3, 123, 45, 1120]
            ],
            objectives=[min, min, min, max, max],
            weights=[0.2941, 0.2353, 0.2353, 0.0588, 0.1765],
            criteria=["C1", "C2", "C3", "C4", "C5"],
        )
        bounds = np.array([
            [14000, 16000],
            [3, 8],
            [80, 140],
            [35, 60],
            [650, 1300]
        ])
        isp = np.array([14000, 3, 80, 60, 1300])

        result = SPOTIS(bounds=bounds, isp=isp).evaluate(dm)
        expected = RankResult(
            "SPOTIS",
            ["A0", "A1", "A2", "A3"],
            [1, 3, 2, 4],
            {"score": [0.4779, 0.5781, 0.5558, 0.5801], "isp": isp, "bounds": bounds}
        )

        self.assert_rank_result(result, expected)

    def test_bounds_from_matrix(self):
        dm = skcriteria.mkdm(
            matrix=[[1, 2, 3], 
                    [4, 5, 6],
                    [10, -1, 4]],
            objectives=[max, min, max],
            weights=[1, 1, 1],
            criteria=["C1", "C2", "C3"]
        )

        # Test that the bounds are calculated from the matrix
        result = SPOTIS().evaluate(dm)
        assert np.all(result.e_.bounds == np.array([[1, 10], [-1, 5], [3, 6]]))
        

    def test_isp_from_bounds(self):
        dm = skcriteria.mkdm(
            matrix=[[1, 2, 3], 
                    [4, 5, 6],
                    [10, -1, 4]],
            objectives=[max, min, max],
            weights=[1, 1, 1],
            criteria=["C1", "C2", "C3"]
        )

        # Test that the ISP is calculated from the bounds given by the matrix
        result = SPOTIS().evaluate(dm)
        assert np.all(result.e_.isp == np.array([10, -1, 6]))

        # Test that the ISP is calculated from the bounds provided
        bounds = np.array([[0, 15], [-5, 11], [3, 6]])
        result = SPOTIS(bounds=bounds).evaluate(dm)
        assert np.all(result.e_.isp == np.array([15, -5, 6]))


    def test_bounds_validation(self):
        dm = skcriteria.mkdm(
            matrix=[[1, 2, 3], 
                    [4, 5, 6],
                    [7, 8, 9]],
            objectives=[max, min, max],
            weights=[1, 1, 1],
            criteria=["C1", "C2", "C3"]
        )
        
        # Note that the bounds are invalid because there is an alternative with a value out of the bounds.
        bounds = np.array([[1, 6], [2, 8], [3, 9]])
        
        with pytest.raises(ValueError, match="The matrix values must be within the provided bounds."):
            SPOTIS(bounds=bounds).evaluate(dm)


    def test_isp_validation(self):
        dm = skcriteria.mkdm(
            matrix=[[1, 2, 3], 
                    [4, 5, 6],
                    [7, 8, 9]],
            objectives=[max, min, max],
            weights=[1, 1, 1],
            criteria=["C1", "C2", "C3"]
        )

        # Note that the ISP is invalid because it is outside the bounds of the criterias.
        bounds = np.array([[1, 7], [2, 8], [3, 9]])
        isp = np.array([0, 2, 3])

        with pytest.raises(ValueError, match="The isp values must be within the provided bounds."):
            SPOTIS(bounds=bounds, isp=isp).evaluate(dm)
        

    def assert_rank_result(self, result, expected, atol=1e-4):
        assert result.values_equals(expected)
        assert result.method == expected.method
        assert np.allclose(result.e_.score, expected.e_.score, atol=atol)
        assert np.all(result.e_.bounds == expected.e_.bounds)
        assert np.all(result.e_.isp == expected.e_.isp)


    
