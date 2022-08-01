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
    """Abstract class capable of impute missing values of the matrix.

    This abstract class require to redefine ``_impute``, instead of
    ``_transform_data``.

    """

    _skcriteria_abstract_class = True

    @abc.abstractmethod
    def _impute(self, matrix):
        """Impute the missing values.

        Parameters
        ----------
        matrix: :py:class:`numpy.ndarray`
            The decision matrix to weights.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The imputed matrix.

        """
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
    """Imputation transformer for completing missing values.

    Internally this class uses

    Parameters
    ----------
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    strategy : str, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0.

    """

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
        """The placeholder for the missing values."""
        return self._missing_values

    @property
    def strategy(self):
        """The imputation strategy."""
        return self._strategy

    @property
    def fill_value(self):
        """ fill_value is used to replace all occurrences of missing_values, \
        when strategy == "constant"."""
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
    """Multivariate imputer that estimates each feature from all the others.
    A strategy for imputing missing values by modeling each feature with
    missing values as a function of other features in a round-robin fashion.
    Read more in the :ref:`User Guide <iterative_imputer>`.
    .. versionadded:: 0.21
    .. note::
      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import `enable_iterative_imputer`::
        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_iterative_imputer  # noqa
        >>> # now you can import normally from sklearn.impute
        >>> from sklearn.impute import IterativeImputer

    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.
    missing_values : int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
    sample_posterior : bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        `return_std` in its `predict` method if set to `True`. Set to
        `True` if using `IterativeImputer` for multiple imputations.
    max_iter : int, default=10
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each criteria with missing values. The stopping criterion
        is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
        where `X_t` is `X` at iteration `t`. Note that early stopping is only
        applied if `sample_posterior=False`.
    tol : float, default=1e-3
        Tolerance of the stopping condition.
    n_nearest_criteria : int, default=None
        Number of other criteria to use to estimate the missing values of
        each feature column. Nearness between criteria is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of criteria throughout the
        imputation process, the neighbor criteria are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of criteria is huge. If `None`, all criteria will be used.
    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, \
            default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.
    imputation_order : {'ascending', 'descending', 'roman', 'arabic', \
            'random'}, default='ascending'
        The order in which the criteria will be imputed. Possible values:
        - `'ascending'`: From criteria with fewest missing values to most.
        - `'descending'`: From criteria with most missing values to fewest.
        - `'roman'`: Left to right.
        - `'arabic'`: Right to left.
        - `'random'`: A random order for each round.
    skip_complete : bool, default=False
        If `True` then criteria with missing values during :meth:`transform`
        which did not have any missing values during :meth:`fit` will be
        imputed with the initial imputation method only. Set to `True` if you
        have many criteria with no missing values at both :meth:`fit` and
        :meth:`transform` time to save compute.
    min_value : float or array-like of shape (n_criteria,), default=-np.inf
        Minimum possible imputed value. Broadcast to shape `(n_criteria,)` if
        scalar. If array-like, expects shape `(n_criteria,)`, one min value for
        each feature. The default is `-np.inf`.
    max_value : float or array-like of shape (n_criteria,), default=np.inf
        Maximum possible imputed value. Broadcast to shape `(n_criteria,)` if
        scalar. If array-like, expects shape `(n_criteria,)`, one max value for
        each feature. The default is `np.inf`.
    verbose : int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.
    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator criteria if `n_nearest_features` is not `None`,
        the `imputation_order` if `random`, and the sampling from posterior if
        `sample_posterior=True`. Use an integer for determinism.
        See :term:`the Glossary <random_state>`.

    """

    _skcriteria_parameters = [
        "estimator",
        "missing_values",
        "sample_posterior",
        "max_iter",
        "tol",
        "n_nearest_criteria",
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
        n_nearest_criteria=None,
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
        self._n_nearest_criteria = n_nearest_criteria
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
    def n_nearest_criteria(self):
        return self._n_nearest_criteria

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
            n_nearest_features=self._n_nearest_criteria,
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
