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
import inspect
from collections import Counter
from functools import partial

import numpy as np

import pandas as pd

from ..core import SKCMethodABC, mkdm
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


def _validate_evaluate_dm_parameter(cls, dm_param):
    """
    Validate that the 'dm' parameter meets the required rules.

    Rules:
    1. Must be named 'dm'
    2. Must be positional
    3. Must not have a default value

    Parameters
    ----------
    cls : class
        The class containing the method
    dm_param : inspect.Parameter
        The 'dm' parameter to validate

    Raises
    ------
    TypeError
        If any of the rules are not met
    """
    # Check decision matrix name
    if dm_param.name != "dm":
        raise TypeError(
            f"First parameter after 'self' in {cls.__name__}.evaluate() "
            f"must be named 'dm', got '{dm_param.name}'"
        )

    # Check it is positional
    if dm_param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(
            f"Parameter 'dm' in {cls.__name__}.evaluate() "
            f"must be positional, got {dm_param.kind.name}"
        )

    # Check it has no default value
    if dm_param.default != inspect.Parameter.empty:
        raise TypeError(
            f"Parameter 'dm' in {cls.__name__}.evaluate() "
            f"must not have a default value"
        )


#: Parameters that are not allowed in the evaluate method
_FORBIDEN_EVALUATE_EXTRA_PARAMS = tuple(
    sorted(inspect.signature(mkdm).parameters)
)


def _validate_evaluate_additional_parameters(cls, params):
    """
    Validate that all additional parameters meet the required rules.

    Rules for additional parameters:
    1. Must have default values
    2. Must be keyword-only parameters

    Parameters
    ----------
    cls : class
        The class containing the method
    params : list of inspect.Parameter
        List of additional parameters to validate

    Raises
    ------
    TypeError
        If any parameter doesn't meet the requirements
    """
    for param in params:
        # Check that parameter has a default value
        if param.default == inspect.Parameter.empty:
            raise TypeError(
                f"Parameter '{param.name}' in {cls.__name__}.evaluate() "
                f"must have a default value"
            )

        # Check that parameter is keyword-only
        if param.kind != inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"Parameter '{param.name}' in {cls.__name__}.evaluate() "
                f"must be keyword-only, got {param.kind.name}"
            )

        # Check that parameter name is not in mkdm function parameters
        if param.name in _FORBIDEN_EVALUATE_EXTRA_PARAMS:
            raise TypeError(
                f"Parameter '{param.name}' in {cls.__name__}.evaluate() "
                "is forbidden. Forbidden names: "
                f"{_FORBIDEN_EVALUATE_EXTRA_PARAMS}"
            )


class SKCDecisionMakerABC(SKCMethodABC):
    """Abstract class for all decisor based methods in scikit-criteria."""

    _skcriteria_abstract_class = True
    _skcriteria_dm_type = "decision_maker"

    def __init_subclass__(cls):
        """Validate if the subclass are well formed.

        This method is called by the metaclass when a subclass is created.

        Raises
        ------
        TypeError
            If the subclass is not well formed

        """
        super().__init_subclass__()

        # Get the evaluate method from the subclass
        evaluate_method = getattr(cls, "evaluate")

        if evaluate_method is not SKCDecisionMakerABC.evaluate:

            sig = inspect.signature(evaluate_method)
            params = list(sig.parameters.values())

            # Skip 'self'
            dm, *additional_params = params[1:]

            # validate the parameters
            _validate_evaluate_dm_parameter(cls, dm)
            _validate_evaluate_additional_parameters(cls, additional_params)

    @abc.abstractmethod
    def _evaluate_data(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_result(self, alternatives, values, extra):
        raise NotImplementedError()

    def _evaluate_dm(self, dm, **kwargs):
        """Internal method that contains the core evaluation logic.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.
        **kwargs:
            Additional parameters

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.

        """
        data = dm.to_dict()

        result_data, extra = self._evaluate_data(**data, **kwargs)

        alternatives = data["alternatives"]
        result = self._make_result(
            alternatives=alternatives,
            values=result_data,
            extra=extra,
        )

        return result

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
        return self._evaluate_dm(dm)


# =============================================================================
# RESULTS
# =============================================================================


class ResultABC(DiffEqualityMixin, metaclass=abc.ABCMeta):
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

    e_ = extra_  # shortcut to extra_

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

    @doc_inherit(DiffEqualityMixin.diff)
    def diff(
        self,
        other,
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False,
        check_dtypes=False,
    ):
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
            "extra_": partial(
                dict_allclose, rtol=rtol, atol=atol, equal_nan=equal_nan
            ),
        }

        the_diff = diff(self, other, **members)
        return the_diff

    def values_equals(self, other):
        """Check if the alternatives and values are the same.

        The method doesn't check the method or the extra parameters.

        """
        the_diff = self.diff(other)
        return (
            "alternatives" not in the_diff.members_diff
            and "values" not in the_diff.members_diff
        )

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
        # the sorted unique values of the rank!
        # [1, 1, 1, 2, 3] >>> [1, 2, 3] <<< OK! this is consecutive
        # [1, 1, 4, 4, 3] >>> [1, 3, 4]  <<< BAD this is not consecutive
        cleaned_values = np.sort(np.unique(values))

        # the size of the sorted unique values
        # len([1, 2, 3]) => 3
        # len([1, 3, 4]) => 3
        length = len(cleaned_values)

        # this create the expected rank of this length (must start in 1)
        # [1, 2, 3] -> [1, 2, 3]
        # [1, 3, 4] -> [1, 2, 3]
        expected = np.arange(length) + 1

        # if the sorted unique values are not the expected, this is not a rank
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
        version="0.7",
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
