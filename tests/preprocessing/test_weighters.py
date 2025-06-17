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
from skcriteria.preprocessing.weighters import (
    CRITIC,
    Critic,
    MEREC,
    EntropyWeighter,
    EqualWeighter,
    SKCWeighterABC,
    StdWeighter,
    critic_weights,
    pearson_correlation,
    spearman_correlation,
    Objective
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


def test_MEREC_():
    """
    Data from:
    """

    matrix = np.array(
        [
            [450, 8000, 54, 145],
            [10, 9100, 2, 160],
            [100, 8200, 31, 153],
            [220, 9300, 1, 162],
            [5, 8400, 23, 158]
        ]
    )
    
    objectives = [max, max, min, min]
    weights = [1, 1, 1, 1]

    where_max = np.array([obj is max for obj in objectives])

    maxs = matrix.max(axis=0)
    mins = matrix.min(axis=0)

    normalized_matrix = np.where(
        where_max,
        mins / matrix,
        matrix / maxs
    )

    dm = skcriteria.mkdm(
        matrix=normalized_matrix,
        objectives=objectives,
        weights=weights
    )

    expected = skcriteria.mkdm(
            matrix=[
                [0.011, 1, 1, 0.895],
                [0.500, 0.879, 0.037, 0.988],
                [0.050, 0.976, 0.574, 0.944],
                [0.023, 0.860, 0.019, 1],
                [1, 0.952, 0.426, 0.975]
            ],
            objectives=[max, max, min, min],
            weights=[0.5752, 0.0141, 0.4016, 0.0091],
    )
    
    weighter = MEREC()
    
    result = weighter.transform(dm)
    assert result.aequals(expected)
