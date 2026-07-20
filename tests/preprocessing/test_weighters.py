#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.weighters"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

import pytest

import scipy

import skcriteria
from skcriteria.preprocessing.scalers import VectorScaler
from skcriteria.preprocessing.weighters import (
    CRITIC,
    Critic,
    EntropyWeighter,
    EqualWeighter,
    GiniWeighter,
    MEREC,
    RANCOM,
    SKCWeighterABC,
    StdWeighter,
    critic_weights,
    pearson_correlation,
    rancom_weights,
    spearman_correlation,
)


# =============================================================================
# WEIGHTER
# =============================================================================


def test_SKCWeighterABC_weight_matrix_not_implemented(decision_matrix):
    class Foo(SKCWeighterABC):
        _skcriteria_parameters = []

        def _weight_matrix(self, **kwargs):
            return super()._weight_matrix(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCWeighterABC_not_redefined_abc_methods():
    class Foo(SKCWeighterABC):
        _skcriteria_parameters = []

    with pytest.raises(TypeError):
        Foo()


def test_SKCWeighterABC_flow(decision_matrix):
    dm = decision_matrix(seed=42)
    expected_weights = np.ones(dm.matrix.shape[1]) * 42

    class Foo(SKCWeighterABC):
        _skcriteria_parameters = []

        def _weight_matrix(self, matrix, **kwargs):
            return expected_weights

    transformer = Foo()

    expected = skcriteria.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=expected_weights,
        dtypes=dm.dtypes,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
    )

    result = transformer.transform(dm)

    assert result.equals(expected)


# =============================================================================
# TEST EQUAL WEIGHTERS
# =============================================================================


def test_EqualWeighter_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 2], [4, 5]],
        objectives=[min, max],
        weights=[1, 2],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2], [4, 5]],
        objectives=[min, max],
        weights=[1, 1],
    )

    weighter = EqualWeighter(base_value=2)

    result = weighter.transform(dm)

    assert result.equals(expected)


def test_EqualWeighter(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    expected = skcriteria.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=[1 / 20] * 20,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    weighter = EqualWeighter()
    result = weighter.transform(dm)

    assert result.equals(expected)


# =============================================================================
# STD
# =============================================================================


def test_StdWeighter_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 2], [4, 16]],
        objectives=[min, max],
        weights=[1, 2],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2], [4, 16]],
        objectives=[min, max],
        weights=[0.176471, 0.82353],
    )

    weighter = StdWeighter()

    result = weighter.transform(dm)
    assert result.aequals(expected, atol=1e-5)


def test_StdWeighter(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    std = np.std(dm.matrix, axis=0, ddof=1)

    expected = skcriteria.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=std / np.sum(std),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    weighter = StdWeighter()
    result = weighter.transform(dm)

    assert result.equals(expected)


# =============================================================================
# STD
# =============================================================================


def test_EntropyWeighter_simple_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 2], [4, 16]],
        objectives=[min, max],
        weights=[1, 2],
    )

    expected = skcriteria.mkdm(
        matrix=[[1, 2], [4, 16]],
        objectives=[min, max],
        weights=[0.358889, 0.641111],
    )

    weighter = EntropyWeighter()
    result = weighter.transform(dm)

    assert result.aequals(expected, atol=1e-5)


def test_EntropyWeighter(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=20,
        max_criteria=20,
        min_objectives_proportion=0.5,
    )

    entropy = 1 - scipy.stats.entropy(dm.matrix, base=10, axis=0)

    expected = skcriteria.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=entropy / np.sum(entropy),
        alternatives=dm.alternatives,
        criteria=dm.criteria,
        dtypes=dm.dtypes,
    )

    weighter = EntropyWeighter()
    result = weighter.transform(dm)

    assert result.equals(expected)


def test_EntropyWeighter_less_predictable_more_weight():
    dm = skcriteria.mkdm(
        [
            [1, 20, 300],
            [1, 20, 400],
            [1, 30, 500],
            [1, 30, 600],
            [1, 40, 700],
            [1, 40, 800],
        ],
        objectives=[max, max, max],
        criteria="C0 C1 C2".split(),
    )

    weighter = EntropyWeighter()
    result = weighter.transform(dm)

    assert result.weights["C0"] < result.weights["C1"]
    assert result.weights["C1"] < result.weights["C2"]


# =============================================================================
# CRITIC
# =============================================================================


def test_CRITIC_diakoulaki1995determining():
    """
    Data from:
        Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        Determining objective weights in multiple criteria problems:
        The critic method. Computers & Operations Research, 22(7), 763-770.
    """

    dm = skcriteria.mkdm(
        matrix=[
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9.0, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4.0, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98],
        ],
        objectives=[max, max, max],
        weights=[61, 1.08, 4.33],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98],
        ],
        objectives=[max, max, max],
        weights=[0.20222554, 0.48090173, 0.31687273],
    )

    weighter = CRITIC()

    result = weighter.transform(dm)
    assert result.aequals(expected)


def test_CRITIC_minimize_warning():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    weighter = CRITIC()

    with pytest.warns(UserWarning):
        weighter.transform(dm)


def test_CRITIC_bad_correlation():
    with pytest.raises(ValueError):
        CRITIC(correlation="foo")
    with pytest.raises(ValueError):
        CRITIC(correlation=1)


@pytest.mark.parametrize(
    "scale, correlation, result",
    [
        (
            True,
            "pearson",
            [0.20222554, 0.48090173, 0.31687273],
        ),
        (
            False,
            "pearson",
            [0.86874234, 0.08341434, 0.04784331],
        ),
        (
            True,
            "spearman",
            [0.21454645, 0.4898563, 0.29559726],
        ),
        (
            False,
            "spearman",
            [0.87672195, 0.08082369, 0.04245436],
        ),
    ],
)
def test_critic_weight_weights_diakoulaki1995determining(
    scale, correlation, result
):
    """
    Data from:
        Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        Determining objective weights in multiple criteria problems:
        The critic method. Computers & Operations Research, 22(7), 763-770.
    """
    mtx = [
        [61, 1.08, 4.33],
        [20.7, 0.26, 4.34],
        [16.3, 1.98, 2.53],
        [9, 3.29, 1.65],
        [5.4, 2.77, 2.33],
        [4, 4.12, 1.21],
        [-6.1, 3.52, 2.10],
        [-34.6, 3.31, 0.98],
    ]
    weights = critic_weights(
        mtx, objectives=[1, 1, 1], scale=scale, correlation=correlation
    )
    assert np.allclose(weights, result)


# deprecated ==================================================================


def test_Critic_deprecation_warning():
    with pytest.deprecated_call():
        Critic()


def test_pearson_correlation_with_deprecation_warning():
    mtx = np.array(
        [
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98],
        ]
    )
    expected = pd.DataFrame(mtx).corr("pearson").to_numpy()

    with pytest.deprecated_call():
        result = pearson_correlation(mtx.T)

    np.testing.assert_allclose(result, expected)


def test_spearman_correlation_with_deprecation_warning():
    mtx = np.array(
        [
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98],
        ]
    )
    expected = pd.DataFrame(mtx).corr("spearman").to_numpy()

    with pytest.deprecated_call():
        result = spearman_correlation(mtx.T)

    np.testing.assert_allclose(result, expected)


# =============================================================================
# MEREC
# =============================================================================


def test_MEREC_keshavarz2021determination():
    """
    Data from:
        Keshavarz-Ghorabaee, M., Amiri, M., Zavadskas, E. K., Turskis, Z.,
        & Antucheviciene, J. (2021).
        Determination of objective weights using a new method based on the
        removal effects of criteria (MEREC).
        Symmetry, 13(4), 525.
    """

    dm = skcriteria.mkdm(
        matrix=[
            [450, 8000, 54, 145],
            [10, 9100, 2, 160],
            [100, 8200, 31, 153],
            [220, 9300, 1, 162],
            [5, 8400, 23, 158],
        ],
        objectives=[max, max, min, min],
    )

    expected = skcriteria.mkdm(
        matrix=[
            [450, 8000, 54, 145],
            [10, 9100, 2, 160],
            [100, 8200, 31, 153],
            [220, 9300, 1, 162],
            [5, 8400, 23, 158],
        ],
        objectives=[max, max, min, min],
        weights=[0.5752, 0.0141, 0.4016, 0.0091],
    )

    weighter = MEREC()

    result = weighter.transform(dm)
    assert result.aequals(expected, atol=1e-3)


# =============================================================================
# Gini
# =============================================================================


def test_Gini_WomenVulnerabilityIndex():
    """
    Data from:
        Aggarwal, S., Aggarwal, G., & Bansal, M. (2024).
        Effect of Different MCDM Techniques and Weighting Mechanisms on
        Women Vulnerability Index. International Journal of Intelligent
        Systems and Applications in Engineering, 12(21s), 3291–3299.
    """
    data = [
        [28, 38, 34, 265, 39, 10, 25, 226, 541],
        [3, 1, 3, 13, 3, 0, 0, 3, 7],
        [12, 17, 106, 153, 28, 8, 12, 77, 223],
        [200, 207, 2373, 1482, 448, 91, 284, 2347, 3325],
        [24, 20, 306, 272, 73, 11, 30, 279, 497],
        [3, 3, 10, 40, 6, 0, 3, 15, 127],
        [26, 39, 276, 356, 67, 13, 19, 261, 488],
        [119, 209, 2800, 2447, 828, 203, 189, 4164, 4862],
        [14, 18, 135, 166, 18, 6, 14, 145, 307],
        [12, 19, 105, 119, 10, 4, 9, 126, 296],
        [62, 57, 773, 556, 148, 44, 73, 769, 1147],
        [42, 93, 474, 603, 59, 11, 33, 372, 1361],
        [10, 25, 112, 203, 16, 1, 10, 132, 458],
        [155, 255, 2256, 1935, 447, 105, 243, 2273, 3183],
        [133, 153, 1141, 1561, 235, 28, 108, 1301, 3329],
        [2, 0, 5, 6, 4, 1, 0, 10, 11],
        [1, 2, 4, 16, 5, 0, 0, 6, 25],
        [0, 1, 2, 5, 1, 0, 0, 4, 3],
        [0, 0, 0, 5, 7, 0, 0, 2, 2],
        [17, 31, 279, 268, 44, 17, 28, 260, 449],
        [37, 68, 827, 688, 116, 23, 32, 780, 1314],
        [235, 319, 2927, 2466, 1168, 212, 306, 4220, 3762],
        [0, 2, 3, 10, 1, 0, 0, 3, 8],
        [68, 56, 453, 696, 75, 19, 51, 815, 1498],
        [30, 40, 304, 298, 59, 6, 17, 253, 633],
        [2, 0, 13, 19, 2, 0, 1, 12, 30],
        [1425, 3830, 34370, 21473, 7234, 2185, 1484, 38174, 54931],
        [87, 114, 1121, 875, 176, 65, 105, 958, 1666],
        [47, 69, 459, 656, 100, 27, 51, 587, 1723],
        [2, 2, 3, 28, 0, 0, 0, 5, 16],
        [3, 20, 88, 159, 21, 2, 5, 105, 182],
        [1, 3, 2, 9, 3, 0, 1, 9, 19],
        [0, 0, 4, 6, 0, 0, 0, 9, 27],
        [0, 0, 1, 2, 0, 0, 0, 1, 4],
        [348, 675, 6164, 5997, 860, 376, 291, 5902, 12087],
        [2, 3, 24, 48, 5, 2, 3, 35, 54],
    ]

    dm = skcriteria.mkdm(
        matrix=data,
        weights=[1] * 9,
        objectives=[max] * 9,
    )

    scaler = VectorScaler(target="matrix")
    dm = scaler.transform(dm)

    weighter = GiniWeighter()
    result = weighter.transform(dm)

    expected = skcriteria.mkdm(
        matrix=data,
        weights=[
            0.1055,
            0.1122,
            0.1131,
            0.1076,
            0.1140,
            0.1169,
            0.1062,
            0.1136,
            0.1104,
        ],
        objectives=[max] * 9,
    )

    expected = scaler.transform(expected)

    assert result.aequals(expected, atol=1e-3)


# =============================================================================
# RANCOM
# =============================================================================


def test_RANCOM_weights_function():
    weights = np.array([6.0, 6.0, 3.5, 0.5, 1.5, 2.5, 4.5])
    expected = np.array(
        [0.2449, 0.2449, 0.1429, 0.0204, 0.0612, 0.1020, 0.1837]
    )

    result = rancom_weights(weights)

    np.testing.assert_allclose(result, expected, atol=1e-3)


def test_RANCOM_weights_with_ties():
    weights = np.array([0.3, 0.3, 0.2, 0.2])

    result = rancom_weights(weights)

    # Check that weights sum to 1
    assert np.isclose(np.sum(result), 1.0, atol=1e-10)
    # Check that all weights are non-negative
    assert np.all(result >= 0)


def test_RANCOM_weighter_ideal_inputs():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
        objectives=[max, max, max, max, max],
        weights=[0.1, 0.2, 0.3, 0.25, 0.15],
    )

    weighter = RANCOM()
    result = weighter.transform(dm)

    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-10)
    assert np.all(result.weights >= 0)
    np.testing.assert_array_equal(result.matrix, dm.matrix)
    np.testing.assert_array_equal(result.objectives, dm.objectives)


def test_RANCOM_weighter_fewer_than_five_weights_warning():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3, 4], [2, 3, 4, 5]],
        objectives=[max, max, max, max],
        weights=[0.25, 0.25, 0.25, 0.25],
    )

    weighter = RANCOM()

    with pytest.warns(
        UserWarning,
        match="RANCOM method proves to be a more suitable solution",
    ):
        result = weighter.transform(dm)

    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-10)
    assert np.all(result.weights >= 0)
