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

import abc
import enum
import functools

from matplotlib import cm, colors

import numpy as np

import pandas as pd

import pyquery as pq

from .utils import Bunch, doc_inherit


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

        self._objectives = np.array(
            [Objective.construct_from_alias(a) for a in objectives]
        )

        self._weights = np.array(weights, dtype=float)

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
        matrix,
        objectives,
        weights=None,
        anames=None,
        cnames=None,
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

        anames: Iterable o None (default ``None``)
            Optional names of the alternatives. If is ``None``,
            al the alternatives are names "A[n]" where n is the number of
            the row of `matrix` statring at 0.

        cnames: Iterable o None (default ``None``)
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

        data_df = pd.DataFrame(matrix, index=anames, columns=cnames)

        if dtypes is not None and len(dtypes) != c_number:
            raise ValueError(f"'dtypes' must have {c_number} elements")
        elif dtypes is not None:
            dtypes = {c: dt for c, dt in zip(cnames, dtypes)}
            data_df = data_df.astype(dtypes)

        return cls(data_df=data_df, objectives=objectives, weights=weights)

    # MCDA ====================================================================

    @property
    def anames(self):
        """Names of the alternatives."""
        return self._data_df.index.to_numpy()

    @property
    def cnames(self):
        """Names of the criteria."""
        return self._data_df.columns.to_numpy()

    @property
    def matrix(self):
        """Alternatives matrix as 2D numpy array."""
        return self._data_df.to_numpy()

    @property
    def weights(self):
        """Weights of the criteria."""
        return np.copy(self._weights)

    @property
    def objectives_values(self):
        """Objectives of the criteria as ``int``.

        - Minimize = Objective.MIN.value
        - Maximize = Objective.MAX.value

        """
        return np.array([o.value for o in self._objectives], dtype=int)

    @property
    def objectives(self):
        """Objectives of the criteria as ``Objective`` instances."""
        return np.copy(self._objectives)

    @property
    def dtypes(self):
        """Dtypes of the criteria."""
        return self._data_df.dtypes.to_numpy()

    # UTILITIES ===============================================================

    def copy(self):
        """Return a deep copy of the current DecisionMatrix."""
        return DecisionMatrix(
            data_df=self._data_df,
            objectives=self._objectives,
            weights=self._weights,
        )

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
        data = np.vstack((self._objectives, self._weights, self.matrix))
        index = np.hstack((["objectives", "weights"], self.anames))
        df = pd.DataFrame(data, index=index, columns=self.cnames, copy=True)
        return df

    def to_dict(self):
        """Return a dict representation of the data."""
        return {
            "matrix": self.matrix,
            "objectives": self.objectives_values,
            "weights": self.weights,
            "anames": self.anames,
            "cnames": self.cnames,
            "dtypes": self.dtypes,
        }

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
        aequals, :py:function:`numpy.isclose`, :py:function:`numpy.all`,
        :py:function:`numpy.any`, :py:function:`numpy.equal`,
        :py:function:`numpy.allclose`.

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
          'cnames', 'anames' or 'objectives' returns ``False``.
        - Next check the 'weights' and the matrix itself using the provided
          tolerance.

        Parameters
        ----------
        other : :py:class:`skcriteria.DecisionMatrix`
            Other instance to compare.
        rtol : float
            The relative tolerance parameter
            (see Notes in :py:function:`numpy.allclose`).
        atol : float
            The absolute tolerance parameter
            (see Notes in :py:function:`numpy.allclose`).
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
        equals, :py:function:`numpy.isclose`, :py:function:`numpy.all`,
        :py:function:`numpy.any`, :py:function:`numpy.equal`,
        :py:function:`numpy.allclose`.

        """
        return (self is other) or (
            isinstance(other, DecisionMatrix)
            and np.shape(self) == np.shape(other)
            and np.array_equal(self.cnames, other.cnames)
            and np.array_equal(self.anames, other.anames)
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
        for c, o, w in zip(self.cnames, self.objectives, self.weights):
            header = f"{c}[{o.to_string()} {w}]"
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

        kwargs = {"header": header, "show_dimensions": False}

        # retrieve the original string
        original_string = self._data_df.to_string(**kwargs)

        # add dimension
        string = f"{original_string}\n[{dimensions}]"

        return string

    def _repr_html_(self):
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


# =============================================================================
# RESULTS
# =============================================================================


class ResultBase(metaclass=abc.ABCMeta):

    _skcriteria_result_column = None

    def __init_subclass__(cls):
        """Validate if the subclass are well formed."""
        result_column = cls._skcriteria_result_column
        if result_column is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_result_column'")

    def __init__(self, method, anames, values, extra):
        self._validate_result(values)
        self._method = str(method)
        self._extra = Bunch("extra", extra)
        self._result_df = pd.DataFrame(
            values, index=anames, columns=[self._skcriteria_result_column]
        )

    @abc.abstractmethod
    def _validate_result(self, values):
        raise NotImplementedError()

    @property
    def values(self):
        return self._result_df[self._skcriteria_result_column].to_numpy()

    @property
    def method(self):
        return self._method

    @property
    def anames(self):
        return self._result_df.index.to_numpy()

    @property
    def extra_(self):
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


class RankResult(ResultBase):

    _skcriteria_result_column = "Rank"

    def _validate_result(self, values):
        length = len(values)
        expected = np.arange(length) + 1
        if not np.array_equal(np.sort(values), expected):
            raise ValueError(f"The data {values} doesn't look like a ranking")

    @property
    def rank_(self):
        return self.values

    def _repr_html_(self):
        """Return a html representation for a particular result.

        Mainly for IPython notebook.
        """

        # retrieve the original string
        df = self._result_df.T
        original_html = df.style.background_gradient(axis=1)._repr_html_()

        # add metadata
        html = (
            "<div class='skcresult-rank skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )

        return html


class KernelResult(ResultBase):

    _skcriteria_result_column = "Kernel"

    def _validate_result(self, values):
        if np.asarray(values).dtype != bool:
            raise ValueError(f"The data {values} doesn't look like a kernel")

    @property
    def kernel_(self):
        return self.values

    @property
    def kernelwhere_(self):
        return np.where(self.kernel_)[0]

    def _repr_html_(self):
        """Return a html representation for a particular result.

        Mainly for IPython notebook.
        """

        cmap = cm.get_cmap("PuBu")
        bool_colors = {
            True: colors.to_hex(cmap.get_over()),
            False: colors.to_hex(cmap.get_under()),
        }

        def color_negative_red(val):
            bg = bool_colors[val]
            fg = bool_colors[not val]
            return f"background-color: {bg}; color: {fg}"

        df = self._result_df.T

        original_html = df.style.applymap(color_negative_red)._repr_html_()

        # add metadata
        html = (
            "<div class='skcresult-kernel skcresult'>\n"
            f"{original_html}"
            f"<em class='skcresult-method'>Method: {self.method}</em>\n"
            "</div>"
        )

        return html
