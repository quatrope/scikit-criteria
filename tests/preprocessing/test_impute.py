#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.impute

"""

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

    result = impute.SimpleImputer().transform(dm)
    expected_mtx = sklimpute.SimpleImputer().fit_transform(dm.matrix)

    assert np.isnan(result.matrix.to_numpy()).sum() == 0
    np.testing.assert_array_equal(result.matrix.to_numpy(), expected_mtx)


def test_SimpleImputer_params_vs_sklearn():
    result = sorted(impute.SimpleImputer._skcriteria_parameters)

    ignore = ["verbose", "add_indicator", "copy"]
    alias = {}
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

    result = impute.IterativeImputer().transform(dm)
    expected_mtx = sklimpute.IterativeImputer().fit_transform(dm.matrix)

    assert np.isnan(result.matrix.to_numpy()).sum() == 0
    np.testing.assert_array_equal(result.matrix.to_numpy(), expected_mtx)


def test_IterativeImputer_params_vs_sklearn():
    from sklearn.experimental import enable_iterative_imputer  # noqa

    result = sorted(impute.IterativeImputer._skcriteria_parameters)

    ignore = ["add_indicator", "skip_complete"]
    alias = {"n_nearest_features": "n_nearest_criteria"}

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

    result = impute.KNNImputer().transform(dm)
    expected_mtx = sklimpute.KNNImputer().fit_transform(dm.matrix)

    assert np.isnan(result.matrix.to_numpy()).sum() == 0
    np.testing.assert_array_equal(result.matrix.to_numpy(), expected_mtx)


def test_KNNImputer_params_vs_sklearn():
    result = sorted(impute.KNNImputer._skcriteria_parameters)

    ignore = ["add_indicator", "copy"]
    alias = {}
    expected = sorted(
        [
            alias.get(p, p)
            for p in sklimpute.KNNImputer().get_params(deep=False)
            if p not in ignore
        ]
    )

    assert result == expected
