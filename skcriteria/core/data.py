#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
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


import enum
import functools
from collections import abc

import numpy as np

import pandas as pd
from pandas.io.formats import format as pd_fmt

import pyquery as pq


from .dominance import DecisionMatrixDominanceAccessor
from .plot import DecisionMatrixPlotter
from .stats import DecisionMatrixStatsAccessor
from ..utils import deprecated, doc_inherit


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
# _SLICER ARRAY
# =============================================================================
class _ACArray(np.ndarray, abc.Mapping):
    """Immutable Array to provide access to the alternative and criteria \
    values.

    The behavior is the same as a numpy.ndarray but if the slice it receives
    is a value contained in the array it uses an external function
    to access the series with that criteria/alternative.

    Besides this it has the typical methods of a dictionary.

    """

    def __new__(cls, input_array, skc_slicer):
        obj = np.asarray(input_array).view(cls)
        obj._skc_slicer = skc_slicer
        return obj

    @doc_inherit(np.ndarray.__getitem__)
    def __getitem__(self, k):
        try:
            if k in self:
                return self._skc_slicer(k)
            return super().__getitem__(k)
        except IndexError:
            raise IndexError(k)

    def __setitem__(self, k, v):
        """Raise an AttributeError, this object are read-only."""
        raise AttributeError("_SlicerArray are read-only")

    @doc_inherit(abc.Mapping.items)
    def items(self):
        return ((e, self[e]) for e in self)

    @doc_inherit(abc.Mapping.keys)
    def keys(self):
        return iter(self)

    @doc_inherit(abc.Mapping.values)
    def values(self):
        return (self[e] for e in self)


# =============================================================================
# DECISION MATRIX
# =============================================================================
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
        :py:class:`pandas.DataFrame` where the index is the alternatives
        and the columns are the criteria; an iterable with the objectives with
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

    Parameters
    ----------
    data_df: :py:class:`pandas.DatFrame`
        Dataframe where the index is the alternatives and the columns
        are the criteria.
    objectives: :py:class:`numpy.ndarray`
        Aan iterable with the targets with sense of optimality of every
        criteria (You can use any alias defined in Objective)
        the same length as columns/criteria has the data_df.
    weights: :py:class:`numpy.ndarray`
        An iterable with the weights also with the same amount of elements
        as criteria.

    """

    def __init__(self, data_df, objectives, weights):

        self._data_df = (
            data_df.copy()
            if isinstance(data_df, pd.DataFrame)
            else pd.DataFrame(data_df)
        )

        self._objectives = np.asarray(objectives, dtype=object)
        self._weights = np.asanyarray(weights, dtype=float)

        if not (
            len(self._data_df.columns)
            == len(self._weights)
            == len(self._objectives)
        ):
            raise ValueError(
                "The number of weights, and objectives must be equal to the "
                "number of criteria (number of columns in data_df)"
            )

    # CUSTOM CONSTRUCTORS =====================================================

    @classmethod
    def from_mcda_data(
        cls,
        matrix,
        objectives,
        weights=None,
        alternatives=None,
        criteria=None,
        dtypes=None,
    ):
        """Create a new DecisionMatrix object.

        This method receives the parts of the matrix, in what conceptually
        the matrix of alternatives is usually divided

        Parameters
        ----------
        matrix: Iterable
            The matrix of alternatives. Where every row is an alternative
            and every column is a criteria.

        objectives: Iterable
            The array with the sense of optimality of every
            criteria. You can use any alias provided by the objective class.

        weights: Iterable o None (default ``None``)
            Optional weights of the criteria. If is ``None`` all the criteria
            are weighted with 1.

        alternatives: Iterable o None (default ``None``)
            Optional names of the alternatives. If is ``None``,
            al the alternatives are names "A[n]" where n is the number of
            the row of `matrix` statring at 0.

        criteria: Iterable o None (default ``None``)
            Optional names of the criteria. If is ``None``,
            al the alternatives are names "C[m]" where m is the number of
            the columns of `matrix` statring at 0.

        dtypes: Iterable o None (default ``None``)
            Optional types of the criteria. If is None, the type is inferred
            automatically by pandas.

        Returns
        -------
        :py:class:`DecisionMatrix`
            A new decision matrix.


        Example
        -------


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

        Notes
        -----
        This functionality generates more sensitive defaults than using the
        constructor of the DecisionMatrix class but is slower.

        """
        # first we need the number of alternatives and criteria
        try:
            a_number, c_number = np.shape(matrix)
        except ValueError:
            matrix_ndim = np.ndim(matrix)
            raise ValueError(
                f"'matrix' must have 2 dimensions, found {matrix_ndim} instead"
            )

        alternatives = np.asarray(
            [f"A{idx}" for idx in range(a_number)]
            if alternatives is None
            else alternatives
        )
        if len(alternatives) != a_number:
            raise ValueError(f"'alternatives' must have {a_number} elements")

        criteria = np.asarray(
            [f"C{idx}" for idx in range(c_number)]
            if criteria is None
            else criteria
        )

        if len(criteria) != c_number:
            raise ValueError(f"'criteria' must have {c_number} elements")

        weights = np.asarray(np.ones(c_number) if weights is None else weights)

        data_df = pd.DataFrame(matrix, index=alternatives, columns=criteria)

        if dtypes is not None and len(dtypes) != c_number:
            raise ValueError(f"'dtypes' must have {c_number} elements")
        elif dtypes is not None:
            dtypes = {c: dt for c, dt in zip(criteria, dtypes)}
            data_df = data_df.astype(dtypes)

        return cls(data_df=data_df, objectives=objectives, weights=weights)

    # MCDA ====================================================================
    #     This properties are usefull to access interactively to the
    #     underlying data a. Except for alternatives and criteria all other
    #     properties expose the data as dataframes or series

    @property
    def alternatives(self):
        """Names of the alternatives."""
        arr = self._data_df.index.to_numpy()
        slicer = self._data_df.loc.__getitem__
        return _ACArray(arr, slicer)

    @property
    def criteria(self):
        """Names of the criteria."""
        arr = self._data_df.columns.to_numpy()
        slicer = self._data_df.__getitem__
        return _ACArray(arr, slicer)

    @property
    def weights(self):
        """Weights of the criteria."""
        return pd.Series(
            self._weights,
            dtype=float,
            index=self._data_df.columns,
            name="Weights",
        )

    @property
    def objectives(self):
        """Objectives of the criteria as ``Objective`` instances."""
        return pd.Series(
            [Objective.construct_from_alias(a) for a in self._objectives],
            index=self._data_df.columns,
            name="Objectives",
        )

    @property
    def minwhere(self):
        """Mask with value True if the criterion is to be minimized."""
        mask = self.objectives == Objective.MIN
        mask.name = "minwhere"
        return mask

    @property
    def maxwhere(self):
        """Mask with value True if the criterion is to be maximized."""
        mask = self.objectives == Objective.MAX
        mask.name = "maxwhere"
        return mask

    # READ ONLY PROPERTIES ====================================================

    @property
    def iobjectives(self):
        """Objectives of the criteria as ``int``.

        - Minimize = Objective.MIN.value
        - Maximize = Objective.MAX.value

        """
        return pd.Series(
            [o.value for o in self.objectives],
            dtype=np.int8,
            index=self._data_df.columns,
        )

    @property
    def matrix(self):
        """Alternatives matrix as pandas DataFrame.

        The matrix excludes weights and objectives.

        If you want to create a DataFrame with objetvies and weights, use
        ``DecisionMatrix.to_dataframe()``

        """
        return self._data_df.copy()

    @property
    def dtypes(self):
        """Dtypes of the criteria."""
        return self._data_df.dtypes.copy()

    # ACCESSORS (YES, WE USE CACHED PROPERTIES IS THE EASIEST WAY) ============

    @property
    @functools.lru_cache(maxsize=None)
    def plot(self):
        """Plot accessor."""
        return DecisionMatrixPlotter(self)

    @property
    @functools.lru_cache(maxsize=None)
    def stats(self):
        """Descriptive statistics accessor."""
        return DecisionMatrixStatsAccessor(self)

    @property
    @functools.lru_cache(maxsize=None)
    def dominance(self):
        """Dominance information accessor."""
        return DecisionMatrixDominanceAccessor(self)

    # UTILITIES ===============================================================

    def copy(self, **kwargs):
        """Return a deep copy of the current DecisionMatrix.

        This method is also useful for manually modifying the values of the
        DecisionMatrix object.

        Parameters
        ----------
        kwargs :
            The same parameters supported by ``from_mcda_data()``. The values
            provided replace the existing ones in the object to be copied.

        Returns
        -------
        :py:class:`DecisionMatrix`
            A new decision matrix.

        """
        dmdict = self.to_dict()
        dmdict.update(kwargs)

        return self.from_mcda_data(**dmdict)

    def to_dataframe(self):
        """Convert the entire DecisionMatrix into a dataframe.

        The objectives and weights ara added as rows before the alternatives.

        Returns
        -------
        :py:class:`pd.DataFrame`
            A Decision matrix as pandas DataFrame.

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
        data = np.vstack((self.objectives, self.weights, self.matrix))
        index = np.hstack((["objectives", "weights"], self.alternatives))
        df = pd.DataFrame(data, index=index, columns=self.criteria, copy=True)
        return df

    def to_dict(self):
        """Return a dict representation of the data.

        All the values are represented as numpy array.
        """
        return {
            "matrix": self.matrix.to_numpy(),
            "objectives": self.iobjectives.to_numpy(),
            "weights": self.weights.to_numpy(),
            "dtypes": self.dtypes.to_numpy(),
            "alternatives": np.asarray(self.alternatives),
            "criteria": np.asarray(self.criteria),
        }

    @deprecated(
        reason=(
            "Use 'DecisionMatrix.stats()', "
            "'DecisionMatrix.stats(\"describe\")' or "
            "'DecisionMatrix.stats.describe()' instead."
        ),
        version=0.6,
    )
    def describe(self, **kwargs):
        """Generate descriptive statistics.

        Descriptive statistics include those that summarize the central
        tendency, dispersion and shape of a dataset's distribution,
        excluding ``NaN`` values.

        Parameters
        ----------
        Same parameters as ``pandas.DataFrame.describe()``.

        Returns
        -------
        ``pandas.DataFrame``
            Summary statistics of DecisionMatrix provided.

        """
        return self._data_df.describe(**kwargs)

    # CMP =====================================================================

    @property
    def shape(self):
        """Return a tuple with (number_of_alternatives, number_of_criteria).

        dm.shape <==> np.shape(dm)

        """
        return np.shape(self._data_df)

    def __len__(self):
        """Return the number ot alternatives.

        dm.__len__() <==> len(dm).

        """
        return len(self._data_df)

    def equals(self, other):
        """Return True if the decision matrix are equal.

        This method calls `DecisionMatrix.aquals` whitout tolerance.

        Parameters
        ----------
        other : :py:class:`skcriteria.DecisionMatrix`
            Other instance to compare.

        Returns
        -------
        equals : :py:class:`bool:py:class:`
            Returns True if the two dm are equals.

        See Also
        --------
        aequals, :py:func:`numpy.isclose`, :py:func:`numpy.all`,
        :py:func:`numpy.any`, :py:func:`numpy.equal`,
        :py:func:`numpy.allclose`.

        """
        return self.aequals(other, 0, 0, False)

    def aequals(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
        """Return True if the decision matrix are equal within a tolerance.

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
        other : :py:class:`skcriteria.DecisionMatrix`
            Other instance to compare.
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
            Returns True if the two dm are equal within the given
            tolerance; False otherwise.

        See Also
        --------
        equals, :py:func:`numpy.isclose`, :py:func:`numpy.all`,
        :py:func:`numpy.any`, :py:func:`numpy.equal`,
        :py:func:`numpy.allclose`.

        """
        return (self is other) or (
            isinstance(other, DecisionMatrix)
            and np.shape(self) == np.shape(other)
            and np.array_equal(self.criteria, other.criteria)
            and np.array_equal(self.alternatives, other.alternatives)
            and np.array_equal(self.objectives, other.objectives)
            and np.allclose(
                self.weights,
                other.weights,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
            and np.allclose(
                self.matrix,
                other.matrix,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )

    # repr ====================================================================
    def _get_cow_headers(self):
        """Columns names with COW (Criteria, Objective, Weight)."""
        headers = []
        fmt_weights = pd_fmt.format_array(self.weights, None)
        for c, o, w in zip(self.criteria, self.objectives, fmt_weights):
            header = f"{c}[{o.to_string()}{w}]"
            headers.append(header)
        return headers

    def _get_axc_dimensions(self):
        """Dimension foote with AxC (Alternativs x Criteria)."""
        a_number, c_number = self.shape
        dimensions = f"{a_number} Alternatives x {c_number} Criteria"
        return dimensions

    def __repr__(self):
        """dm.__repr__() <==> repr(dm)."""
        header = self._get_cow_headers()
        dimensions = self._get_axc_dimensions()

        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        max_cols = pd.get_option("display.max_columns")
        max_colwidth = pd.get_option("display.max_colwidth")

        width = (
            pd.io.formats.console.get_console_size()[0]
            if pd.get_option("display.expand_frame_repr")
            else None
        )

        original_string = self._data_df.to_string(
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            line_width=width,
            max_colwidth=max_colwidth,
            show_dimensions=False,
            header=header,
        )

        # add dimension
        string = f"{original_string}\n[{dimensions}]"

        return string

    def _repr_html_(self):
        """Return a html representation for a particular DecisionMatrix.

        Mainly for IPython notebook.
        """
        header = dict(zip(self.criteria, self._get_cow_headers()))
        dimensions = self._get_axc_dimensions()

        # retrieve the original string
        with pd.option_context("display.show_dimensions", False):
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
            th.text = header.get(crit, crit)

        return str(d)


# =============================================================================
# factory
# =============================================================================


@functools.wraps(DecisionMatrix.from_mcda_data)
def mkdm(*args, **kwargs):
    """Alias for DecisionMatrix.from_mcda_data."""
    return DecisionMatrix.from_mcda_data(*args, **kwargs)
