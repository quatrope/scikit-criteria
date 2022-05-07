#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
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
from ..utils import Bunch, deprecated, doc_inherit

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

    @abc.abstractmethod
    def _make_result(self, alternatives, values, extra):
        raise NotImplementedError()

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


class ResultABC(metaclass=abc.ABCMeta):
    """Base class to implement different types of results.

    Any evaluation of the DecisionMatrix is expected to result in an object
    that extends the functionalities of this class.

    Parameters
    ----------
    method: str
        Name of the method that generated the result.
    alternatives: array-like
        Names of the alternatives evaluated.
    values: array-like
        Values assigned to each alternative by the method, where the i-th
        value refers to the valuation of the i-th. alternative.
    extra: dict-like
        Extra information provided by the method regarding the evaluation of
        the alternatives.

    """

    _skcriteria_result_column = None

    def __init_subclass__(cls):
        """Validate if the subclass are well formed."""
        result_column = cls._skcriteria_result_column
        if result_column is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_result_column'")

    def __init__(self, method, alternatives, values, extra):
        self._validate_result(values)
        self._method = str(method)
        self._extra = Bunch("extra", extra)
        self._result_df = pd.DataFrame(
            values,
            index=alternatives,
            columns=[self._skcriteria_result_column],
        )

    @abc.abstractmethod
    def _validate_result(self, values):
        """Validate that the values are the expected by the result type."""
        raise NotImplementedError()

    @property
    def values(self):
        """Values assigned to each alternative by the method.

        The i-th value refers to the valuation of the i-th. alternative.

        """
        return self._result_df[self._skcriteria_result_column].to_numpy()

    @property
    def method(self):
        """Name of the method that generated the result."""
        return self._method

    @property
    def alternatives(self):
        """Names of the alternatives evaluated."""
        return self._result_df.index.to_numpy()

    @property
    def extra_(self):
        """Additional information about the result.

        Note
        ----
        ``e_`` is an alias for this property

        """
        return self._extra

    e_ = extra_

    # CMP =====================================================================

    @property
    def shape(self):
        """Tuple with (number_of_alternatives, number_of_alternatives).

        rank.shape <==> np.shape(rank)

        """
        return np.shape(self._result_df)

    def __len__(self):
        """Return the number ot alternatives.

        rank.__len__() <==> len(rank).

        """
        return len(self._result_df)

    def equals(self, other):
        """Check if the alternatives and ranking are the same.

        The method doesn't check the method or the extra parameters.

        """
        return (self is other) or (
            isinstance(other, RankResult)
            and self._result_df.equals(other._result_df)
        )

    # REPR ====================================================================

    def __repr__(self):
        """result.__repr__() <==> repr(result)."""
        kwargs = {"show_dimensions": False}

        # retrieve the original string
        df = self._result_df.T
        original_string = df.to_string(**kwargs)

        # add dimension
        string = f"{original_string}\n[Method: {self.method}]"

        return string


@doc_inherit(ResultABC, warn_class=False)
class RankResult(ResultABC):
    """Ranking of alternatives.

    This type of results is used by methods that generate a ranking of
    alternatives.

    """

    _skcriteria_result_column = "Rank"

    @doc_inherit(ResultABC._validate_result)
    def _validate_result(self, values):

        cleaned_values = np.unique(values)

        length = len(cleaned_values)
        expected = np.arange(length) + 1
        if not np.array_equal(np.sort(cleaned_values), expected):
            raise ValueError(f"The data {values} doesn't look like a ranking")

    @property
    def has_ties_(self):
        """Return True if two alternatives shares the same ranking."""
        values = self.values
        return len(np.unique(values)) != len(values)

    @property
    def ties_(self):
        """Counter object that counts how many times each value appears."""
        return Counter(self.values)

    @property
    def rank_(self):
        """Alias for ``values``."""
        return self.values

    @property
    def untied_rank_(self):
        """Ranking whitout ties.

        if the ranking has ties this property assigns unique and consecutive
        values in the ranking. This method only assigns the values using the
        command ``numpy.argsort(rank_) + 1``.

        """
        if self.has_ties_:
            return np.argsort(self.rank_) + 1
        return self.rank_

    def _repr_html_(self):
        """Return a html representation for a particular result.

        Mainly for IPython notebook.

        """
        df = self._result_df.T
        original_html = df.style._repr_html_()

        # add metadata
        html = (
            "<div class='skcresult-rank skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )

        return html


@doc_inherit(ResultABC, warn_class=False)
class KernelResult(ResultABC):
    """Separates the alternatives between good (kernel) and bad.

    This type of results is used by methods that select which alternatives
    are good and bad. The good alternatives are called "kernel"

    """

    _skcriteria_result_column = "Kernel"

    @doc_inherit(ResultABC._validate_result)
    def _validate_result(self, values):
        if np.asarray(values).dtype != bool:
            raise ValueError(f"The data {values} doesn't look like a kernel")

    @property
    def kernel_(self):
        """Alias for ``values``."""
        return self.values

    @property
    def kernel_size_(self):
        """How many alternatives has the kernel."""
        return np.sum(self.kernel_)

    @property
    def kernel_where_(self):
        """Indexes of the alternatives that are part of the kernel."""
        return np.where(self.kernel_)[0]

    @property
    @deprecated(
        reason=("Use 'kernel_where_' instead"),
        version=0.7,
    )
    def kernelwhere_(self):
        """Indexes of the alternatives that are part of the kernel."""
        return self.kernel_where_

    @property
    def kernel_alternatives_(self):
        """Return the names of alternatives in the kernel."""
        return self._result_df.index[self._result_df.Kernel].to_numpy()

    def _repr_html_(self):
        """Return a html representation for a particular result.

        Mainly for IPython notebook.

        """
        df = self._result_df.T
        original_html = df._repr_html_()

        # add metadata
        html = (
            "<div class='skcresult-kernel skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )

        return html
