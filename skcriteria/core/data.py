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
the alternative matrix, weights and objectives (MIN, MAX) of the criteria.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import functools
from collections import abc

import numpy as np

import pandas as pd
from pandas.io.formats import format as pd_fmt

from .dominance import DecisionMatrixDominanceAccessor
from .objectives import Objective
from .plot import DecisionMatrixPlotter
from .stats import DecisionMatrixStatsAccessor
from ..utils import deprecated, df_temporal_header, doc_inherit


# =============================================================================
# SLICERS ARRAY
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


class _Loc:
    """Locator abstraction.

    this class ensures that the correct objectives and weights are applied to
    the sliced ``DecisionMatrix``.

    """

    def __init__(self, name, real_loc, objectives, weights):
        self._name = name
        self._real_loc = real_loc
        self._objectives = objectives
        self._weights = weights

    @property
    def name(self):
        """The name of the locator."""
        return self._name

    def __getitem__(self, slc):
        """dm[slc] <==> dm.__getitem__(slc)."""
        df = self._real_loc.__getitem__(slc)
        if isinstance(df, pd.Series):
            df = df.to_frame().T

            dtypes = self._real_loc.obj.dtypes
            dtypes = dtypes[dtypes.index.isin(df.columns)]

            df = df.astype(dtypes)

        objectives = self._objectives
        objectives = objectives[objectives.index.isin(df.columns)].to_numpy()

        weights = self._weights
        weights = weights[weights.index.isin(df.columns)].to_numpy()

        return DecisionMatrix(df, objectives, weights)


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
            data_df.copy(deep=True)
            if isinstance(data_df, pd.DataFrame)
            else pd.DataFrame(data_df, copy=True)
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
        """Names of the alternatives.

        From this array you can also access the values of the alternatives as
        ``pandas.Series``.

        """
        arr = self._data_df.index.to_numpy(copy=True)
        slicer = self._data_df.loc.__getitem__
        return _ACArray(arr, slicer)

    @property
    def criteria(self):
        """Names of the criteria.

        From this array you can also access the values of the criteria as
        ``pandas.Series``.

        """
        arr = self._data_df.columns.to_numpy(copy=True)
        slicer = self._data_df.__getitem__
        return _ACArray(arr, slicer)

    @property
    def weights(self):
        """Weights of the criteria."""
        return pd.Series(
            self._weights,
            dtype=float,
            index=self._data_df.columns.copy(deep=True),
            name="Weights",
            copy=True,
        )

    @property
    def objectives(self):
        """Objectives of the criteria as ``Objective`` instances."""
        return pd.Series(
            [Objective.from_alias(a) for a in self._objectives],
            index=self._data_df.columns,
            name="Objectives",
            copy=True,
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
            index=self._data_df.columns.copy(deep=True),
            copy=True,
        )

    @property
    def matrix(self):
        """Alternatives matrix as pandas DataFrame.

        The matrix excludes weights and objectives.

        If you want to create a DataFrame with objectives and weights, use
        ``DecisionMatrix.to_dataframe()``

        """
        mtx = self._data_df.copy(deep=True)
        mtx.index = self._data_df.index.copy(deep=True)
        mtx.index.name = "Alternatives"
        mtx.columns = self._data_df.columns.copy(deep=True)
        mtx.columns.name = "Criteria"
        return mtx

    @property
    def dtypes(self):
        """Dtypes of the criteria."""
        series = self._data_df.dtypes.copy(deep=True)
        series.index = self._data_df.dtypes.index.copy(deep=True)
        return series

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
            "Use ``DecisionMatrix.stats()``, "
            "``DecisionMatrix.stats('describe)`` or "
            "``DecisionMatrix.stats.describe()`` instead."
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

        This method calls `DecisionMatrix.aquals` without tolerance.

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

    # SLICES ==================================================================

    def __getitem__(self, slc):
        """dm[slc] <==> dm.__getitem__(slc)."""
        df = self._data_df.__getitem__(slc)
        if isinstance(df, pd.Series):
            df = df.to_frame()

            dtypes = self._data_df.dtypes
            dtypes = dtypes[dtypes.index.isin(df.columns)]

            df = df.astype(dtypes)

        objectives = self.objectives
        objectives = objectives[objectives.index.isin(df.columns)].to_numpy()

        weights = self.weights
        weights = weights[weights.index.isin(df.columns)].to_numpy()

        return DecisionMatrix(df, objectives, weights)

    @property
    def loc(self):
        """Access a group of alternatives and criteria by label(s) or a \
        boolean array.

        ``.loc[]`` is primarily alternative label based, but may also be used
        with a boolean array.

        Unlike DataFrames, `ìloc`` of ``DecisionMatrix`` always returns an
        instance of ``DecisionMatrix``.

        """
        return _Loc("loc", self._data_df.loc, self.objectives, self.weights)

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.

        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean
        array.

        Unlike DataFrames, `ìloc`` of ``DecisionMatrix`` always returns an
        instance of ``DecisionMatrix``.

        """
        return _Loc("iloc", self._data_df.iloc, self.objectives, self.weights)

    # REPR ====================================================================

    def _get_cow_headers(
        self, only=None, fmt="{criteria}[{objective}{weight}]"
    ):
        """Columns names with COW (Criteria, Objective, Weight)."""
        criteria = self._data_df.columns
        objectives = self.objectives
        weights = self.weights

        if only:
            mask = self._data_df.columns.isin(only)
            criteria = criteria[mask]
            objectives = objectives[mask]
            weights = weights[mask]

        weights = pd_fmt.format_array(weights, None)

        headers = []
        for crit, obj, weight in zip(criteria, objectives, weights):
            header = fmt.format(
                criteria=crit, objective=obj.to_symbol(), weight=weight
            )
            headers.append(header)
        return np.array(headers)

    def _get_axc_dimensions(self):
        """Dimension foote with AxC (Alternativs x Criteria)."""
        a_number, c_number = self.shape
        dimensions = f"{a_number} Alternatives x {c_number} Criteria"
        return dimensions

    def __repr__(self):
        """dm.__repr__() <==> repr(dm)."""
        header = self._get_cow_headers()
        dimensions = self._get_axc_dimensions()

        with df_temporal_header(self._data_df, header) as df:
            with pd.option_context("display.show_dimensions", False):
                original_string = repr(df)

        # add dimension
        string = f"{original_string}\n[{dimensions}]"

        return string

    def _repr_html_(self):
        """Return a html representation for a particular DecisionMatrix.

        Mainly for IPython notebook.
        """
        header = self._get_cow_headers()
        dimensions = self._get_axc_dimensions()

        # retrieve the original string
        with df_temporal_header(self._data_df, header) as df:
            with pd.option_context("display.show_dimensions", False):
                original_html = df._repr_html_()

        # add dimension
        html = (
            "<div class='decisionmatrix'>\n"
            f"{original_html}"
            f"<em class='decisionmatrix-dim'>{dimensions}</em>\n"
            "</div>"
        )

        return html


# =============================================================================
# factory
# =============================================================================


@functools.wraps(DecisionMatrix.from_mcda_data)
def mkdm(*args, **kwargs):
    """Alias for DecisionMatrix.from_mcda_data."""
    return DecisionMatrix.from_mcda_data(*args, **kwargs)
