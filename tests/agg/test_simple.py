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
    WeightedProductModel,
    WeightedSumModel,
    WASPASModel,
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


def test_paper_expected_results_waspas_lambda_zero_behaves_like_wpm():
    dm = skcriteria.mkdm(
        matrix=[
            [
                0.8486,
                0.6364,
                0.7982,
                0.6707,
                1.0000,
                0.8534,
                0.6622,
                0.8618,
                0.1432,
                0.1585,
                1.0000,
                0.4531,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                0.7976,
                0.7000,
                0.9005,
                0.9324,
                0.6788,
                1.0000,
                0.6500,
                0.7270,
                0.7346,
            ],
            [
                0.6542,
                0.7000,
                0.9169,
                0.8951,
                0.6000,
                0.9791,
                0.6216,
                0.9479,
                0.1340,
                0.1585,
                0.9795,
                0.6728,
            ],
            [
                0.3694,
                0.4375,
                0.9407,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.5478,
                1.0000,
                0.3754,
                1.0000,
            ],
        ],
        objectives=[max] * 12,
        weights=[
            0.0627,
            0.0508,
            0.1114,
            0.0874,
            0.0625,
            0.1183,
            0.0784,
            0.0984,
            0.0530,
            0.1417,
            0.0798,
            0.0557,
        ],
    )

    paper_expected_results_with_lambda_zero = RankResult(
        "WeightedProductModel",
        ["A0", "A1", "A2", "A3"],
        [4, 1, 3, 2],
        {"score": [0.4912, 0.8173, 0.5873, 0.8015]},
    )

    ranker = WeightedProductModel()

    result = ranker.evaluate(dm)

    assert result.values_equals(paper_expected_results_with_lambda_zero)
    assert result.method == paper_expected_results_with_lambda_zero.method
    assert np.allclose(
        10**result.e_.score, paper_expected_results_with_lambda_zero.e_.score
    )


def test_paper_expected_results_waspas_lambda_one_behaves_like_wsm():
    dm = skcriteria.mkdm(
        matrix=[
            [
                0.8486,
                0.6364,
                0.7982,
                0.6707,
                1.0000,
                0.8534,
                0.6622,
                0.8618,
                0.1432,
                0.1585,
                1.0000,
                0.4531,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                0.7976,
                0.7000,
                0.9005,
                0.9324,
                0.6788,
                1.0000,
                0.6500,
                0.7270,
                0.7346,
            ],
            [
                0.6542,
                0.7000,
                0.9169,
                0.8951,
                0.6000,
                0.9791,
                0.6216,
                0.9479,
                0.1340,
                0.1585,
                0.9795,
                0.6728,
            ],
            [
                0.3694,
                0.4375,
                0.9407,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.5478,
                1.0000,
                0.3754,
                1.0000,
            ],
        ],
        objectives=[max] * 12,
        weights=[
            0.0627,
            0.0508,
            0.1114,
            0.0874,
            0.0625,
            0.1183,
            0.0784,
            0.0984,
            0.0530,
            0.1417,
            0.0798,
            0.0557,
        ],
    )

    paper_expected_results_with_lambda_zero = RankResult(
        "WeightedSumModel",
        ["A0", "A1", "A2", "A3"],
        [4, 2, 3, 1],
        {"score": [0.6120, 0.8292, 0.6972, 0.8520]},
    )

    ranker = WeightedSumModel()
    result = ranker.evaluate(dm)
    assert result.values_equals(paper_expected_results_with_lambda_zero)
    assert result.method == paper_expected_results_with_lambda_zero.method
    assert np.allclose(
        result.e_.score,
        paper_expected_results_with_lambda_zero.e_.score,
        atol=1e-4,
    )


def test_waspas_lambda_zero_behaves_like_wpm():
    """
    Test WASPAS with l=0 behaves like WeightedProductModel.
    """
    seed = 42
    dm = _random_waspas_inputs(seed)

    ranker_wpm = WeightedProductModel()
    result_wpm = ranker_wpm.evaluate(dm)

    ranker_waspas = WASPASModel(l=0)
    result_waspas = ranker_waspas.evaluate(dm)

    assert result_waspas.values_equals(result_wpm)
    assert np.allclose(
        result_waspas.e_.score, 10**result_wpm.e_.score, atol=1e-4
    )


def test_waspas_lambda_one_behaves_like_wsm():
    """
    Test WASPAS with l=1 behaves like WeightedSumModel.
    """
    seed = 42
    dm = _random_waspas_inputs(seed)

    ranker_wsm = WeightedSumModel()
    result_wsm = ranker_wsm.evaluate(dm)

    ranker_waspas = WASPASModel(l=1)
    result_waspas = ranker_waspas.evaluate(dm)

    assert result_waspas.values_equals(result_wsm)
    assert np.allclose(result_waspas.e_.score, result_wsm.e_.score, atol=1e-4)


def _random_waspas_inputs(n_alt=4, n_crit=5, cost_ratio=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dm = skcriteria.mkdm(
        matrix=np.random.rand(n_alt, n_crit) * 100,
        objectives=np.random.choice(
            [min, max], size=n_crit, p=[cost_ratio, 1 - cost_ratio]
        ),
        weights=np.random.rand(n_crit),
    )

    transformers = [
        InvertMinimize(),
        SumScaler(target="both"),
    ]
    for t in transformers:
        dm = t.transform(dm)

    return dm


def test_WASPASModel_example():
    """
    Test WASPAS with normalized matrix and known result.

    Reference: Zavadskas, E. K. et al. (2012). Optimization of weighted aggregated sum product assessment.
    Elektronika ir elektrotechnika, 122(6), 3–6.
    """
    matrix = [
        [30, 23, 5, 0.745, 0.745, 1500, 5000],
        [18, 13, 15, 0.745, 0.745, 1300, 6000],
        [15, 12, 10, 0.500, 0.500, 950, 7000],
        [25, 20, 13, 0.745, 0.745, 1200, 4000],
        [14, 18, 14, 0.255, 0.745, 950, 3500],
        [17, 15, 9, 0.745, 0.500, 1250, 5250],
        [23, 18, 20, 0.500, 0.745, 1100, 3000],
        [16, 8, 14, 0.255, 0.500, 1500, 3000],
    ]
    objectives = [max, max, max, max, max, min, min]
    weights = [0.1181, 0.1181, 0.0445, 0.1181, 0.2861, 0.2861, 0.0445]

    def _linear_normalization(x, cost=False):
        if cost:
            return np.min(x) / x
        return x / np.max(x)

    norm_matrix = np.asarray(matrix, dtype=float)
    for i, obj in enumerate(objectives):
        is_cost = obj is min
        norm_matrix[:, i] = _linear_normalization(
            norm_matrix[:, i], cost=is_cost
        )

    dm = skcriteria.mkdm(
        matrix=norm_matrix,
        objectives=[max] * 7,
        weights=weights,
    )

    expected = RankResult(
        "WASPASModel",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        [3, 5, 7, 1, 4, 6, 2, 8],
        {
            "score": [
                0.8329,
                0.7884,
                0.6987,
                0.8831,
                0.7971,
                0.7036,
                0.8728,
                0.5749,
            ]
        },
    )

    ranker = WASPASModel(l=0.5)
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=1e-4)


def test_WASPASModel_with_minimize_fails():
    """WASPAS should raise ValueError if input matrix contains min objectives."""
    dm = skcriteria.mkdm(
        matrix=[[1, 7, 3], [3, 5, 6]],
        objectives=[max, min, max],
    )
    ranker = WASPASModel()

    with pytest.raises(
        ValueError, match="WASPASModel can't operate with minimize objective"
    ):
        ranker.evaluate(dm)


def test_WASPASModel_with_zero_fails():
    """WASPAS should raise ValueError if matrix contains 0s (division/log problems)."""
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 0, 6]],
        objectives=[max, max, max],
    )
    ranker = WASPASModel()

    with pytest.raises(
        ValueError, match="WASPASModel can't operate with values <= 0"
    ):
        ranker.evaluate(dm)


def test_WASPASModel_invalid_l_values():
    """WASPAS should raise ValueError if l is not in [0, 1]"""
    invalid_l_values = [-0.1, 1.1, 2, -5]

    dm = skcriteria.mkdm(
        matrix=[[1, 2], [3, 4]],
        objectives=[max, max],
    )

    for invalid_l in invalid_l_values:
        with pytest.raises(
            ValueError, match="l must be a value between 0 and 1"
        ):
            ranker = WASPASModel(l=invalid_l)
            ranker.evaluate(dm)
