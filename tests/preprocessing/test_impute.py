#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.impute"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria.preprocessing import impute

from sklearn import impute as sklimpute


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_SKCImputerABC__impute_not_implemented(decision_matrix):
    class Foo(impute.SKCImputerABC):
        _skcriteria_parameters = []

        def _impute(self, **kwargs):
            return super()._impute(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SimpleImputer(decision_matrix):
    dm = decision_matrix(seed=42, nan_proportion=0.3)
    assert np.isnan(dm.matrix.to_numpy()).sum() > 0

    imputer = impute.SimpleImputer()
    assert imputer.missing_values is np.nan
    assert imputer.strategy == "mean"
    assert imputer.fill_value is None
    assert imputer.keep_empty_criteria is False

    result = imputer.transform(dm)

    expected_mtx = sklimpute.SimpleImputer().fit_transform(dm.matrix)

    assert np.isnan(result.matrix.to_numpy()).sum() == 0
    np.testing.assert_array_equal(result.matrix.to_numpy(), expected_mtx)


def test_SimpleImputer_params_vs_sklearn():
    result = sorted(impute.SimpleImputer._skcriteria_parameters)

    ignore = ["verbose", "add_indicator", "copy"]
    alias = {"keep_empty_features": "keep_empty_criteria"}
    expected = sorted(
        [
            alias.get(p, p)
            for p in sklimpute.SimpleImputer().get_params(deep=False)
            if p not in ignore
        ]
    )

    assert result == expected


def test_IterativeImputer(decision_matrix):
    from sklearn.experimental import enable_iterative_imputer  # noqa

    dm = decision_matrix(seed=42, nan_proportion=0.3)
    assert np.isnan(dm.matrix.to_numpy()).sum() > 0

    imputer = impute.IterativeImputer()
    assert imputer.imputation_order == "ascending"
    assert imputer.initial_strategy == "mean"
    assert imputer.min_value == -np.inf
    assert imputer.n_nearest_criteria is None
    assert imputer.max_value == np.inf
    assert imputer.verbose == 0
    assert imputer.max_iter == 10
    assert imputer.random_state is None
    assert imputer.tol == 0.001
    assert imputer.missing_values is np.nan
    assert imputer.sample_posterior is False
    assert imputer.estimator is None
    assert imputer.fill_value is None
    assert imputer.keep_empty_criteria is False

    result = imputer.transform(dm)
    expected_mtx = sklimpute.IterativeImputer().fit_transform(dm.matrix)

    assert np.isnan(result.matrix.to_numpy()).sum() == 0
    np.testing.assert_array_equal(result.matrix.to_numpy(), expected_mtx)


def test_IterativeImputer_params_vs_sklearn():
    from sklearn.experimental import enable_iterative_imputer  # noqa

    result = sorted(impute.IterativeImputer._skcriteria_parameters)

    ignore = ["add_indicator", "skip_complete"]
    alias = {
        "n_nearest_features": "n_nearest_criteria",
        "keep_empty_features": "keep_empty_criteria",
    }

    expected = sorted(
        [
            alias.get(p, p)
            for p in sklimpute.IterativeImputer().get_params(deep=False)
            if p not in ignore
        ]
    )
    assert result == expected


def test_KNNImputer(decision_matrix):
    dm = decision_matrix(seed=42, nan_proportion=0.3)
    assert np.isnan(dm.matrix.to_numpy()).sum() > 0

    imputer = impute.KNNImputer()
    assert imputer.missing_values is np.nan
    assert imputer.n_neighbors == 5
    assert imputer.weights == "uniform"
    assert imputer.metric == "nan_euclidean"
    assert imputer.keep_empty_criteria is False

    result = imputer.transform(dm)
    expected_mtx = sklimpute.KNNImputer().fit_transform(dm.matrix)

    assert np.isnan(result.matrix.to_numpy()).sum() == 0
    np.testing.assert_array_equal(result.matrix.to_numpy(), expected_mtx)


def test_KNNImputer_params_vs_sklearn():
    result = sorted(impute.KNNImputer._skcriteria_parameters)

    ignore = ["add_indicator", "copy"]
    alias = {"keep_empty_features": "keep_empty_criteria"}
    expected = sorted(
        [
            alias.get(p, p)
            for p in sklimpute.KNNImputer().get_params(deep=False)
            if p not in ignore
        ]
    )

    assert result == expected
