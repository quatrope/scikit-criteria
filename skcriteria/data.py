#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Data abstraction layer.

This module defines the DecisionMatrix object, which internally encompasses
the alternative matrix,   weights and objectives (MIN, MAX) of the criteria.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import enum
import functools
from typing import Iterable, Optional

import attr

import numpy as np

import pandas as pd

import pyquery as pq


# =============================================================================
# CONSTANTS
# =============================================================================
class Objective(enum.Enum):
    """Representation of criteria objectives (Minimize, Maximize)."""

    #: Internal representation of minimize criteria
    MIN = -1

    #: Internal representation of maximize criteria
    MAX = 1

    # INTERNALS ===============================================================

    _MIN_STR = "\u25bc"
    _MAX_STR = "\u25b2"

    #: Another way to name the maximization criteria.
    _MAX_ALIASES = frozenset(
        [
            MAX,
            _MAX_STR,
            max,
            np.max,
            np.nanmax,
            np.amax,
            "max",
            "maximize",
            "+",
            ">",
        ]
    )

    #: Another ways to name the minimization criteria.
    _MIN_ALIASES = frozenset(
        [
            MIN,
            _MIN_STR,
            min,
            np.min,
            np.nanmin,
            np.amin,
            "min",
            "minimize",
            "<",
            "-",
        ]
    )

    # CUSTOM CONSTRUCTOR ======================================================

    @classmethod
    def construct_from_alias(cls, alias):
        """Return the alias internal representation of the objective."""
        if isinstance(alias, cls):
            return alias
        if isinstance(alias, str):
            alias = alias.lower()
        if alias in cls._MAX_ALIASES.value:
            return cls.MAX
        if alias in cls._MIN_ALIASES.value:
            return cls.MIN
        raise ValueError(f"Invalid criteria objective {alias}")

    # METHODS =================================================================

    def __str__(self):
        """Convert the objective to an string."""
        return self.name

    def to_string(self):
        """Return the printable representation of the objective."""
        if self.value in Objective._MIN_ALIASES.value:
            return Objective._MIN_STR.value
        if self.value in Objective._MAX_ALIASES.value:
            return Objective._MAX_STR.value


# =============================================================================
# DATA CLASS
# =============================================================================

# converters
def _as_df(df):
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def _as_objective_array(arr):
    return np.array([Objective.construct_from_alias(a) for a in arr])


def _as_float_array(arr):
    return np.array(arr, dtype=float)


@attr.s(frozen=True, repr=False, cmp=False)
class DecisionMatrix:
    """Representation of all data needed in the MCDA analysis.

    This object gathers everything necessary to represent a data set used
    in MCDA:

     - An alternative matrix where each row is an alternative and each
       column is of a different criteria.
     - An optimization objective (Minimize, Maximize) for each criterion.
     - A weight for each criterion.
     - An independent type of data for each criterion

     DecisionMatrix has two main forms of construction:

     1. Use the default constructor of the DecisionMatrix class
        ``pandas.DataFrame`` where the index is the alternatives
        and the columns are the criteria; an iterable with the targets with
        the same amount of elements that columns/criteria has the dataframe;
        and an iterable with the weights also with the same amount of elements
        as criteria.

        .. code-block:: pycon

        >>> import pandas as pd
        >>> from skcriteria import DecisionMatrix, mkdm

        >>> data_df = pd.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6]],
        ...     index=["A0", "A1"],
        ...     columns=["C0", "C1", "C2"]
        ... )
        >>> objectives = [min, max, min]
        >>> weights = [1, 1, 1]

        >>> dm = DecisionMatrix(data_df, objectives, weights)
        >>> dm
           C0[▼ 1.0] C1[▲ 1.0] C2[▲ 1.0]
        A0         1         2         3
        A1         4         5         6
        [2 Alternatives x 3 Criteria]

    2. Use the classmethod `DecisionMatrix.from_mcda_data` which requests the
       data in a more natural way for this type of analysis
       (the weights, the criteria / alternative names, and the data types
       are optional)

       >>> DecisionMatrix.from_mcda_data(
       ...     [[1, 2, 3], [4, 5, 6]],
       ...     [min, max, min],
       ...     [1, 1, 1])
          C0[▼ 1.0] C1[▲ 1.0] C2[▲ 1.0]
       A0         1         2         3
       A1         4         5         6
       [2 Alternatives x 3 Criteria]

        For simplicity a function is offered at the module level analogous to
        ``from_mcda_data`` called ``mkdm`` (make decision matrix).

    """

    _data_df: pd.DataFrame = attr.ib(converter=_as_df)
    _objectives: np.ndarray = attr.ib(converter=_as_objective_array)
    _weights: np.ndarray = attr.ib(converter=_as_float_array)

    def __attrs_post_init__(self):
        """Execute the last shape validation.

        Check if the number of columns in in the `data_df` are the same as in
        the `objectives` and `weights`.

        """
        lens = {
            "c_number": len(self._data_df.columns),
            "objectives": len(self._objectives),
            "weights": len(self._weights),
        }
        if len(set(lens.values())) > 1:
            c_number = lens.pop("c_number")
            raise ValueError(
                "'objectives' and 'weights' must have the same number of "
                f"columns in 'data_df {c_number}. Found {lens}."
            )

    # CUSTOM CONSTRUCTORS =====================================================

    @classmethod
    def from_mcda_data(
        cls,
        mtx: Iterable[float],
        objectives: Iterable,
        weights: Optional[Iterable] = None,
        anames: Optional[Iterable] = None,
        cnames: Optional[Iterable] = None,
        dtypes: Optional[Iterable] = None,
    ):
        # first we need the number of alternatives and criteria
        try:
            a_number, c_number = np.shape(mtx)
        except ValueError:
            mtx_ndim = np.ndim(mtx)
            raise ValueError(
                f"'mtx' must have 2 dimensions, found {mtx_ndim} instead"
            )

        anames = np.asarray(
            [f"A{idx}" for idx in range(a_number)]
            if anames is None
            else anames
        )
        if len(anames) != a_number:
            raise ValueError(f"'anames' must have {a_number} elements")

        cnames = np.asarray(
            [f"C{idx}" for idx in range(c_number)]
            if cnames is None
            else cnames
        )

        if len(cnames) != c_number:
            raise ValueError(f"'cnames' must have {c_number} elements")

        weights = np.asarray(np.ones(c_number) if weights is None else weights)

        data_df = pd.DataFrame(mtx, index=anames, columns=cnames)

        if dtypes is not None and len(dtypes) != c_number:
            raise ValueError(f"'dtypes' must have {c_number} elements")
        elif dtypes is not None:
            dtypes = {c: dt for c, dt in zip(cnames, dtypes)}
            data_df = data_df.astype(dtypes)

        return cls(data_df=data_df, objectives=objectives, weights=weights)

    # MCDA ====================================================================

    @property
    def anames(self) -> np.ndarray:
        """Names of the alternatives."""
        return self._data_df.index.to_numpy()

    @property
    def cnames(self) -> np.ndarray:
        """Names of the criteria."""
        return self._data_df.columns.to_numpy()

    @property
    def mtx(self) -> np.ndarray:
        """Alternatives matrix as 2D numpy array."""
        return self._data_df.to_numpy()

    @property
    def weights(self) -> np.ndarray:
        """Weights of the criteria."""
        return np.copy(self._weights)

    @property
    def objectives_values(self) -> np.ndarray:
        """Objectives of the criteria as ``int``.

        - Minimize = Objective.MIN.value
        - Maximize = Objective.MAX.value

        """
        return np.array([o.value for o in self._objectives], dtype=int)

    @property
    def objectives(self) -> np.ndarray:
        """Objectives of the criteria as ``Objective`` instances."""
        return np.copy(self._objectives)

    @property
    def dtypes(self) -> np.ndarray:
        """Dtypes of the criteria."""
        return self._data_df.dtypes.to_numpy()

    # UTILITIES ===============================================================

    def copy(self) -> DecisionMatrix:
        """Return a deep copy of the current DecisionMatrix."""
        return DecisionMatrix(
            data_df=self._data_df,
            objectives=self._objectives,
            weights=self._weights,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the entire DecisionMatrix into a dataframe.

        The objectives and weights ara added as rows before the alternatives.

        Example
        -------
        .. code-block:: pycon

           >>> dm = DecisionMatrix.from_mcda_data(
           >>> dm
           ...     [[1, 2, 3], [4, 5, 6]],
           ...     [min, max, min],
           ...     [1, 1, 1])
               C0[▼ 1.0] C1[▲ 1.0] C2[▲ 1.0]
           A0         1         2         3
           A1         4         5         6

           >>> dm.to_dataframe()
                       C0   C1   C2
           objectives  MIN  MAX  MIN
           weights     1.0  1.0  1.0
           A0            1    2    3
           A1            4    5    6

        """
        data = np.vstack((self._objectives, self._weights, self.mtx))
        index = np.hstack((["objectives", "weights"], self.anames))
        df = pd.DataFrame(data, index=index, columns=self.cnames, copy=True)
        return df

    # CMP =====================================================================

    def __eq__(self, other):
        """dm.__eq__(other) <==> dm == other."""
        return (
            isinstance(other, DecisionMatrix)
            and self._data_df.equals(other._data_df)
            and np.array_equal(self._objectives, other._objectives)
            and np.array_equal(self._weights, other._weights)
        )

    def __ne__(self, other):
        """dm.__ne__(other) <==> dm != other."""
        return not self == other

    # repr ====================================================================
    def _get_cow_headers(self):
        """Columns names with COW (Criteria, Objective, Weight)."""
        headers = []
        for c, o, w in zip(self.cnames, self.objectives, self.weights):
            header = f"{c}[{o.to_string()} {w}]"
            headers.append(header)
        return headers

    def _get_axc_dimensions(self):
        """Dimension foote with AxC (Alternativs x Criteria)."""
        a_number, c_number = np.shape(self._data_df)
        dimensions = f"{a_number} Alternatives x {c_number} Criteria"
        return dimensions

    def __repr__(self) -> str:
        """dm.__repr__() <==> repr(dm)."""
        header = self._get_cow_headers()
        dimensions = self._get_axc_dimensions()

        kwargs = {"header": header, "show_dimensions": False}

        # retrieve the original string
        original_string = self._data_df.to_string(**kwargs)

        # add dimension
        string = f"{original_string}\n[{dimensions}]"

        return string

    def _repr_html_(self) -> str:
        """Return a html representation for a particular DecisionMatrix.

        Mainly for IPython notebook.
        """
        header = dict(zip(self.cnames, self._get_cow_headers()))
        dimensions = self._get_axc_dimensions()

        # retrieve the original string
        original_html = self._data_df._repr_html_()

        # add dimension
        html = (
            "<div class='decisionmatrix'>\n"
            f"{original_html}"
            f"<em class='decisionmatrix-dim'>{dimensions}</em>\n"
            "</div>"
        )

        # now we need to change the table header
        d = pq.PyQuery(html)
        for th in d("div.decisionmatrix table.dataframe > thead > tr > th"):
            crit = th.text
            if crit:
                th.text = header[crit]

        return str(d)


# =============================================================================
# factory
# =============================================================================


@functools.wraps(DecisionMatrix.from_mcda_data)
def mkdm(*args, **kwargs):
    """Alias for DecisionMatrix.from_mcda_data."""
    return DecisionMatrix.from_mcda_data(*args, **kwargs)
