#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities to create madm decision-maker classes."""


# =============================================================================
# imports
# =============================================================================

import abc
from collections import Counter

import numpy as np

import pandas as pd

from ..core import SKCMethodABC
from ..utils import (
    Bunch,
    DiffEqualityMixin,
    deprecated,
    dict_allclose,
    diff,
    doc_inherit,
)

# =============================================================================
# DM BASE
# =============================================================================


class SKCDecisionMakerABC(SKCMethodABC):
    """Abstract class for all decisor based methods in scikit-criteria."""

    _skcriteria_abstract_class = True
    _skcriteria_dm_type = "decision_maker"

    @abc.abstractmethod
    def _evaluate_data(self, **kwargs):
        raise NotImplementedError()

    def _make_result(self, alternatives, values, extra):
        method_name = self.get_method_name()
        return SKCRankResult(
            method=method_name,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def evaluate(self, dm):
        """Validate the dm and calculate and evaluate the alternatives.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.

        """
        data = dm.to_dict()

        result_data, extra = self._evaluate_data(**data)

        alternatives = data["alternatives"]
        result = self._make_result(
            alternatives=alternatives, values=result_data, extra=extra
        )

        return result


# =============================================================================
# RESULTS
# =============================================================================


class SKCRankResult(DiffEqualityMixin):
    """Unified class for all types of decision-making results.

    This class handles rankings of alternatives. All results are treated as
    rankings with consecutive integer values starting from 1.

    Parameters
    ----------
    method: str
        Name of the method that generated the result.
    alternatives: array-like
        Names of the alternatives evaluated.
    values: array-like
        Values assigned to each alternative by the method, where the i-th
        value refers to the valuation of the i-th alternative.
    extra: dict-like
        Extra information provided by the method regarding the evaluation of
        the alternatives.
    """

    def __init__(self, method, alternatives, values, extra=None):
        self._validate_result(values)
        self._method = str(method)
        self._extra = Bunch("extra", extra or {})
        self._result_series = pd.Series(
            values,
            index=pd.Index(alternatives, name="Alternatives", copy=True),
            name="Result",
            copy=True,
        )

    @classmethod
    def from_kernel(cls, method, alternatives, values, extra=None):
        """Create a Result from a kernel mask.

        Parameters
        ----------
        method: str
            Name of the method that generated the result.
        alternatives: array-like
            Names of the alternatives evaluated.
        values: array-like of bool
            Boolean mask where True indicates good alternatives (kernel)
            and False indicates bad alternatives.
        extra: dict-like, optional
            Extra information provided by the method.

        Returns
        -------
        Result
            Result instance with kernel alternatives having value 1
            and non-kernel alternatives having value 2.
        """
        mask = np.asarray(values, dtype=bool)
        # Convert True to 1 (good) and False to 2 (bad)
        values = np.where(mask, 1, 2)
        return cls(method, alternatives, values, extra)

    def _validate_result(self, values):
        """Validate that the values form a valid ranking."""
        values = np.asarray(values)

        # Check if values are numeric
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"The data {values} must be numeric")

        # Get sorted unique values
        cleaned_values = np.sort(np.unique(values))
        length = len(cleaned_values)

        # Create expected consecutive ranking starting from 1
        expected = np.arange(length) + 1

        # Validate that we have consecutive integers starting from 1
        if not np.array_equal(cleaned_values, expected):
            raise ValueError(
                f"The data {values} doesn't form a valid ranking "
                "must be consecutive integers starting from 1)"
            )

    @property
    def values(self):
        """Values assigned to each alternative by the method."""
        return self._result_series.to_numpy(copy=True)

    @property
    def method(self):
        """Name of the method that generated the result."""
        return self._method

    @property
    def alternatives(self):
        """Names of the alternatives evaluated."""
        return self._result_series.index.to_numpy(copy=True)

    @property
    def extra_(self):
        """Additional information about the result."""
        return self._extra

    e_ = extra_  # shortcut to extra_

    # RANKING PROPERTIES
    @property
    def rank_(self):
        """Alias for values (ranking perspective)."""
        return self.values

    @property
    def has_ties_(self):
        """Return True if two alternatives share the same ranking."""
        values = self.values
        return len(np.unique(values)) != len(values)

    @property
    def ties_(self):
        """Counter object that counts how many times each value appears."""
        return Counter(self.values)

    @property
    def untied_rank_(self):
        """Ranking without ties using argsort."""
        if self.has_ties_:
            return np.argsort(self.rank_) + 1
        return self.rank_

    # KERNEL METHODS AND DEPRECATED PROPERTIES

    def kernel(self, max_in_kernel=1):
        """Boolean mask indicating kernel alternatives.

        Parameters
        ----------
        max_in_kernel : int, default=1
            Maximum rank value to consider as kernel.
            For binary kernels (1,2), use default value 1.
            For multi-level rankings, can specify higher values.

        Returns
        -------
        numpy.ndarray
            Boolean array where True indicates alternatives in the kernel.
        """
        return self.values <= max_in_kernel

    @property
    @deprecated(
        reason=("Use kernel() method instead"),
        version="0.9",
    )
    def kernel_(self):
        """Boolean mask indicating kernel alternatives (value=1)."""
        return self.kernel()

    def kernel_size(self, max_in_kernel=1):
        """Number of alternatives in the kernel.

        Parameters
        ----------
        max_in_kernel : int, default=1
            Maximum rank value to consider as kernel.

        Returns
        -------
        int
            Count of alternatives in the kernel.
        """
        return np.sum(self.kernel(max_in_kernel))

    @property
    @deprecated(
        reason=("Use kernel_size() method instead"),
        version="0.9",
    )
    def kernel_size_(self):
        """Number of alternatives in the kernel."""
        return self.kernel_size()

    def kernel_where(self, max_in_kernel=1):
        """Indexes of alternatives that are part of the kernel.

        Parameters
        ----------
        max_in_kernel : int, default=1
            Maximum rank value to consider as kernel.

        Returns
        -------
        numpy.ndarray
            Array of indices for kernel alternatives.
        """
        return np.where(self.kernel(max_in_kernel))[0]

    @property
    @deprecated(
        reason=("Use kernel_where() method instead"),
        version="0.9",
    )
    def kernel_where_(self):
        """Indexes of alternatives that are part of the kernel."""
        return self.kernel_where()

    @property
    @deprecated(
        reason=("Use kernelwhere() instead"),
        version="0.7",
    )
    def kernelwhere_(self):
        """Indexes of the alternatives that are part of the kernel."""
        return self.kernel_where_

    def kernel_alternatives(self, max_in_kernel=1):
        """Names of alternatives in the kernel.

        Parameters
        ----------
        max_in_kernel : int, default=1
            Maximum rank value to consider as kernel.

        Returns
        -------
        numpy.ndarray
            Array of alternative names in the kernel.
        """
        mask = self.kernel(max_in_kernel)
        return self._result_series.index[mask].to_numpy(copy=True)

    @property
    @deprecated(
        reason=("Use kernel_alternatives() method instead"),
        version="0.9",
    )
    def kernel_alternatives_(self):
        """Names of alternatives in the kernel."""
        return self.kernel_alternatives()

    # UTILS
    def to_series(self, *, untied=False):
        """Convert result to pandas Series."""
        if untied:
            return pd.Series(
                self.untied_rank_,
                index=self._result_series.index.copy(deep=True),
                copy=True,
                name="Untied Rank",
            )
        return self._result_series.copy(deep=True)

    @property
    def shape(self):
        """Tuple with (number_of_alternatives,)."""
        return np.shape(self._result_series)

    def __len__(self):
        """Return the number of alternatives."""
        return len(self._result_series)

    @doc_inherit(DiffEqualityMixin.diff)
    def diff(
        self,
        other,
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False,
        check_dtypes=False,
    ):
        """Compare this result with another result."""

        def array_allclose(left_value, right_value):
            return np.allclose(
                left_value,
                right_value,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )

        members = {
            "method": np.array_equal,
            "alternatives": np.array_equal,
            "values": array_allclose,
            "extra_": dict_allclose,
        }

        return diff(self, other, **members)

    def values_equals(self, other):
        """Check if alternatives and values are the same."""
        the_diff = self.diff(other)
        return (
            "alternatives" not in the_diff.members_diff
            and "values" not in the_diff.members_diff
        )

    # REPRESENTATION
    def to_string(self):
        """Convert result to string representation.

        Returns
        -------
        str
            String representation of the result.
        """
        kwargs = {"show_dimensions": False}

        df = self._result_series.to_frame().T
        df.columns = ["Rank"]
        original_string = df.to_string(**kwargs)

        string = f"{original_string}\n[Method: {self.method}]"
        return string

    def to_html(self):
        """Convert result to HTML representation.

        Returns
        -------
        str
            HTML representation of the result.
        """
        df = self._result_series.to_frame().T
        df.columns = ["Rank"]
        original_html = df.style._repr_html_()

        html = (
            f"<div class='skcresult-rank skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )
        return html

    def to_kernel_string(self, max_in_kernel=1):
        """Convert result to kernel string representation.

        Parameters
        ----------
        max_in_kernel : int, default=1
            Maximum rank value to consider as kernel.

        Returns
        -------
        str
            String representation showing kernel membership.
        """
        kernel_values = self.kernel(max_in_kernel)
        kernel_series = pd.Series(
            kernel_values, index=self._result_series.index, name="Kernel"
        )

        kwargs = {"show_dimensions": False}
        df = kernel_series.to_frame().T
        original_string = df.to_string(**kwargs)

        string = f"{original_string}\n[Method: {self.method}]"
        return string

    def to_kernel_html(self, max_in_kernel=1):
        """Convert result to kernel HTML representation.

        Parameters
        ----------
        max_in_kernel : int, default=1
            Maximum rank value to consider as kernel.

        Returns
        -------
        str
            HTML representation showing kernel membership.
        """
        kernel_values = self.kernel(max_in_kernel)
        kernel_series = pd.Series(
            kernel_values, index=self._result_series.index, name="Kernel"
        )

        df = kernel_series.to_frame().T
        original_html = df.style._repr_html_()

        html = (
            f"<div class='skcresult-kernel skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )
        return html

    def __repr__(self):
        """String representation of the result."""
        return self.to_string()

    def _repr_html_(self):
        """HTML representation for Jupyter notebooks."""
        return self.to_html()
