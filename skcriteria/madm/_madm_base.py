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

    _skcriteria_result_series = None

    def __init_subclass__(cls):
        """Validate if the subclass are well formed."""
        result_column = cls._skcriteria_result_series
        if result_column is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_result_series'")

    def __init__(self, method, alternatives, values, extra):
        self._validate_result(values)
        self._method = str(method)
        self._extra = Bunch("extra", extra)
        self._result_series = pd.Series(
            values,
            index=pd.Index(alternatives, name="Alternatives", copy=True),
            name=self._skcriteria_result_series,
            copy=True,
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
        """Additional information about the result.

        Note
        ----
        ``e_`` is an alias for this property

        """
        return self._extra

    e_ = extra_

    # UTILS ===================================================================

    def to_series(self):
        """The result as `pandas.Series`."""
        series = self._result_series.copy(deep=True)
        series.index = self._result_series.index.copy(deep=True)
        return series

    # CMP =====================================================================

    @property
    def shape(self):
        """Tuple with (number_of_alternatives, ).

        rank.shape <==> np.shape(rank)

        """
        return np.shape(self._result_series)

    def __len__(self):
        """Return the number ot alternatives.

        rank.__len__() <==> len(rank).

        """
        return len(self._result_series)

    def values_equals(self, other):
        """Check if the alternatives and ranking are the same.

        The method doesn't check the method or the extra parameters.

        """
        return (self is other) or (
            isinstance(other, type(self))
            and self._result_series.equals(other._result_series)
        )

    def aequals(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """Return True if the result are equal within a tolerance.

        The tolerance values are positive, typically very small numbers.  The
        relative difference (`rtol` * abs(`b`)) and the absolute difference
        `atol` are added together to compare against the absolute difference
        between `a` and `b`.

        NaNs are treated as equal if they are in the same place and if
        ``equal_nan=True``.  Infs are treated as equal if they are in the same
        place and of the same sign in both arrays.

        The proceeds as follows:

        - If ``other`` is the same object return ``True``.
        - If ``other`` is not instance of 'DecisionMatrix', has different shape
          'criteria', 'alternatives' or 'objectives' returns ``False``.
        - Next check the 'weights' and the matrix itself using the provided
          tolerance.

        Parameters
        ----------
        other : Result
            Other result to compare.
        rtol : float
            The relative tolerance parameter
            (see Notes in :py:func:`numpy.allclose`).
        atol : float
            The absolute tolerance parameter
            (see Notes in :py:func:`numpy.allclose`).
        equal_nan : bool
            Whether to compare NaN's as equal.  If True, NaN's in dm will be
            considered equal to NaN's in `other` in the output array.

        Returns
        -------
        aequals : :py:class:`bool:py:class:`
            Returns True if the two result are equal within the given
            tolerance; False otherwise.

        See Also
        --------
        equals, :py:func:`numpy.isclose`, :py:func:`numpy.all`,
        :py:func:`numpy.any`, :py:func:`numpy.equal`,
        :py:func:`numpy.allclose`.

        """
        if self is other:
            return True
        is_veq = self.values_equals(other) and set(self._extra) == set(
            other._extra
        )
        keys = set(self._extra)
        while is_veq and keys:
            k = keys.pop()
            sv = self._extra[k]
            ov = other._extra[k]
            if isinstance(ov, np.ndarray):
                is_veq = is_veq and np.allclose(
                    sv,
                    ov,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                )
            else:
                is_veq = is_veq and sv == ov
        return is_veq

    def equals(self, other):
        """Return True if the results are equal.

        This method calls `aquals` without tolerance.

        Parameters
        ----------
        other : :py:class:`skcriteria.DecisionMatrix`
            Other instance to compare.

        Returns
        -------
        equals : :py:class:`bool:py:class:`
            Returns True if the two results are equals.

        See Also
        --------
        aequals, :py:func:`numpy.isclose`, :py:func:`numpy.all`,
        :py:func:`numpy.any`, :py:func:`numpy.equal`,
        :py:func:`numpy.allclose`.

        """
        return self.aequals(other, 0, 0, False)

    def __eq__(self, other):
        """x.__eq__(y) <==> x == y."""
        return self.equals(other)

    def __ne__(self, other):
        """x.__eq__(y) <==> x == y."""
        return not self == other

    # REPR ====================================================================

    def __repr__(self):
        """result.__repr__() <==> repr(result)."""
        kwargs = {"show_dimensions": False}

        # retrieve the original string
        df = self._result_series.to_frame().T
        original_string = df.to_string(**kwargs)

        # add dimension
        string = f"{original_string}\n[Method: {self.method}]"

        return string

    def _repr_html_(self):
        """Return a html representation for a particular result.

        Mainly for IPython notebook.

        """
        df = self._result_series.to_frame().T
        original_html = df.style._repr_html_()
        rtype = self._skcriteria_result_series.lower()

        # add metadata
        html = (
            f"<div class='skcresult-{rtype} skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )

        return html


@doc_inherit(ResultABC, warn_class=False)
class RankResult(ResultABC):
    """Ranking of alternatives.

    This type of results is used by methods that generate a ranking of
    alternatives.

    """

    _skcriteria_result_series = "Rank"

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

    def to_series(self, *, untied=False):
        """The result as `pandas.Series`."""
        if untied:
            return pd.Series(
                self.untied_rank_,
                index=self._result_series.index.copy(deep=True),
                copy=True,
                name="Untied rank",
            )
        return super().to_series()


@doc_inherit(ResultABC, warn_class=False)
class KernelResult(ResultABC):
    """Separates the alternatives between good (kernel) and bad.

    This type of results is used by methods that select which alternatives
    are good and bad. The good alternatives are called "kernel"

    """

    _skcriteria_result_series = "Kernel"

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
        reason=("Use ``kernel_where_`` instead"),
        version=0.7,
    )
    def kernelwhere_(self):
        """Indexes of the alternatives that are part of the kernel."""
        return self.kernel_where_

    @property
    def kernel_alternatives_(self):
        """Return the names of alternatives in the kernel."""
        return self._result_series.index[self._result_series].to_numpy(
            copy=True
        )
