#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Module that provides multiple strategies for missing value imputation.

The classes implemented here are a thin layer on top of the `sklearn.impute`
module classes.

"""


# =============================================================================
# IMPORTS
# =============================================================================


import abc

import numpy as np

import sklearn.impute as _sklimpute

from ._preprocessing_base import SKCTransformerABC
from ..utils import doc_inherit

# =============================================================================
# BASE CLASS
# =============================================================================


class SKCImputerABC(SKCTransformerABC):

    _skcriteria_abstract_class = True

    @abc.abstractmethod
    def _impute(self, matrix):
        raise NotImplementedError()

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, **kwargs):
        imputed_matrix = self._impute(matrix=matrix)
        kwargs.update(matrix=imputed_matrix, dtypes=None)
        return kwargs


# =============================================================================
# SIMPLE IMPUTER
# =============================================================================


class SimpleImputer(SKCImputerABC):

    _skcriteria_parameters = ["missing_values", "strategy", "fill_value"]

    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
    ):
        self._missing_values = missing_values
        self._strategy = strategy
        self._fill_value = fill_value

    # PROPERTIES ==============================================================

    @property
    def missing_values(self):
        return self._missing_values

    @property
    def strategy(self):
        return self._strategy

    @property
    def fill_value(self):
        return self._fill_value

    # THE IMPUTATION LOGIC ====================================================

    @doc_inherit(SKCImputerABC._impute)
    def _impute(self, matrix):
        imputer = _sklimpute.SimpleImputer(
            missing_values=self._missing_values,
            strategy=self._strategy,
            fill_value=self._fill_value,
        )
        imputed_matrix = imputer.fit_transform(matrix)
        return imputed_matrix


# =============================================================================
# MULTIVARIATE
# =============================================================================


class IterativeImputer(SKCImputerABC):

    _skcriteria_parameters = [
        "estimator",
        "missing_values",
        "sample_posterior",
        "max_iter",
        "tol",
        "n_nearest_features",
        "initial_strategy",
        "imputation_order",
        "skip_complete",
        "min_value",
        "max_value",
        "verbose",
        "random_state",
    ]

    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
    ):
        self._estimator = estimator
        self._missing_values = missing_values
        self._sample_posterior = sample_posterior
        self._max_iter = max_iter
        self._tol = tol
        self._n_nearest_features = n_nearest_features
        self._initial_strategy = initial_strategy
        self._imputation_order = imputation_order
        self._skip_complete = skip_complete
        self._min_value = min_value
        self._max_value = max_value
        self._verbose = verbose
        self._random_state = random_state

    # PROPERTIES ==============================================================

    @property
    def estimator(self):
        return self._estimator

    @property
    def missing_values(self):
        return self._missing_values

    @property
    def sample_posterior(self):
        return self._sample_posterior

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def tol(self):
        return self._tol

    @property
    def n_nearest_features(self):
        return self._n_nearest_features

    @property
    def initial_strategy(self):
        return self._initial_strategy

    @property
    def imputation_order(self):
        return self._imputation_order

    @property
    def skip_complete(self):
        return self._skip_complete

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def verbose(self):
        return self._verbose

    @property
    def random_state(self):
        return self._random_state

    # THE IMPUTATION LOGIC ====================================================

    @doc_inherit(SKCImputerABC._impute)
    def _impute(self, matrix):
        from sklearn.experimental import enable_iterative_imputer  # noqa

        imputer = _sklimpute.IterativeImputer(
            estimator=self._estimator,
            missing_values=self._missing_values,
            sample_posterior=self._sample_posterior,
            max_iter=self._max_iter,
            tol=self._tol,
            n_nearest_features=self._n_nearest_features,
            initial_strategy=self._initial_strategy,
            imputation_order=self._imputation_order,
            skip_complete=self._skip_complete,
            min_value=self._min_value,
            max_value=self._max_value,
            verbose=self._verbose,
            random_state=self._random_state,
        )
        imputed_matrix = imputer.fit_transform(matrix)
        return imputed_matrix


# =============================================================================
# KNN
# =============================================================================


class KNNImputer(SKCImputerABC):

    _skcriteria_parameters = [
        "missing_values",
        "n_neighbors",
        "weights",
        "metric",
    ]

    def __init__(
        self,
        *,
        missing_values=np.nan,
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
    ):
        self._missing_values = missing_values
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._metric = metric

    # PROPERTIES ==============================================================

    @property
    def missing_values(self):
        return self._missing_values

    @property
    def n_neighbors(self):
        return self._n_neighbors

    @property
    def weights(self):
        return self._weights

    @property
    def metric(self):
        return self._metric

    # THE IMPUTATION LOGIC ====================================================

    @doc_inherit(SKCImputerABC._impute)
    def _impute(self, matrix):
        imputer = _sklimpute.KNNImputer(
            missing_values=self._missing_values,
            n_neighbors=self._n_neighbors,
            weights=self._weights,
            metric=self._metric,
        )
        imputed_matrix = imputer.fit_transform(matrix)
        return imputed_matrix
