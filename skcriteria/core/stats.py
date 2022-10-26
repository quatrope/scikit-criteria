#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Stats helper for the DecisionMatrix object."""


# =============================================================================
# IMPORTS
# =============================================================================k

from ..utils import AccessorABC

# =============================================================================
# STATS ACCESSOR
# =============================================================================


class DecisionMatrixStatsAccessor(AccessorABC):
    """Calculate basic statistics of the decision matrix.

    Kind of statistic to produce:

    - 'corr' : Compute pairwise correlation of columns, excluding
      NA/null values.
    - 'cov' : Compute pairwise covariance of columns, excluding NA/null
      values.
    - 'describe' : Generate descriptive statistics.
    - 'kurtosis' : Return unbiased kurtosis over requested axis.
    - 'mad' : Return the mean absolute deviation of the values over the
      requested axis.
    - 'max' : Return the maximum of the values over the requested axis.
    - 'mean' : Return the mean of the values over the requested axis.
    - 'median' : Return the median of the values over the requested
      axis.
    - 'min' : Return the minimum of the values over the requested axis.
    - 'pct_change' : Percentage change between the current and a prior
      element.
    - 'quantile' : Return values at the given quantile over requested
      axis.
    - 'sem' : Return unbiased standard error of the mean over requested
      axis.
    - 'skew' : Return unbiased skew over requested axis.
    - 'std' : Return sample standard deviation over requested axis.
    - 'var' : Return unbiased variance over requested axis.

    """

    # The list of methods that can be accessed of the subjacent dataframe.
    _DF_WHITELIST = (
        "corr",
        "cov",
        "describe",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "pct_change",
        "quantile",
        "sem",
        "skew",
        "std",
        "var",
    )

    _default_kind = "describe"

    def __init__(self, dm):
        self._dm = dm

    def __getattr__(self, a):
        """x.__getattr__(a) <==> x.a <==> getattr(x, "a")."""
        if a not in self._DF_WHITELIST:
            raise AttributeError(a)
        return getattr(self._dm._data_df, a)

    def __dir__(self):
        """x.__dir__() <==> dir(x)."""
        return super().__dir__() + [
            e for e in dir(self._dm._data_df) if e in self._DF_WHITELIST
        ]

    def mad(self, axis=0, skipna=True):
        """Return the mean absolute deviation of the values over a given axis.

        Parameters
        ----------
        axis : int
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.

        """
        df = self._dm._data_df
        return (df - df.mean(axis=axis)).abs().mean(axis=axis, skipna=skipna)
