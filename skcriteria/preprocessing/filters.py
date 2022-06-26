#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Normalization through the distance to distance function."""


# =============================================================================
# IMPORTS
# =============================================================================

import abc
from collections.abc import Collection

import numpy as np

from ._preprocessing_base import SKCTransformerABC
from ..core import DecisionMatrix
from ..utils import doc_inherit

# =============================================================================
# BASE CLASS
# =============================================================================


class SKCByCriteriaFilterABC(SKCTransformerABC):
    """Abstract class capable of filtering alternatives based on criteria \
    values.

    This abstract class require to redefine ``_coerce_filters`` and
    ``_make_mask``, instead of ``_transform_data``.

    Parameters
    ----------
    criteria_filters: dict
        It is a dictionary in which the key is the name of a criterion, and
        the value is the filter condition.
    ignore_missing_criteria: bool, default: False
        If True, it is ignored if a decision matrix does not have any
        particular criteria that should be filtered.

    """

    _skcriteria_parameters = ["criteria_filters", "ignore_missing_criteria"]

    _skcriteria_abstract_class = True

    def __init__(self, criteria_filters, *, ignore_missing_criteria=False):
        if not len(criteria_filters):
            raise ValueError("Must provide at least one filter")
        self._criteria, self._filters = self._coerce_filters(criteria_filters)
        self._ignore_missing_criteria = bool(ignore_missing_criteria)

    @property
    def criteria_filters(self):
        """Conditions on which the alternatives will be evaluated.

        It is a dictionary in which the key is the name of a
        criterion, and the value is the filter condition.

        """
        return dict(zip(self._criteria, self._filters))

    @property
    def ignore_missing_criteria(self):
        """If the value is True the filter ignores the lack of a required \
        criterion.

        If the value is False, the lack of a criterion causes the filter to
        fail.

        """
        return self._ignore_missing_criteria

    @abc.abstractmethod
    def _coerce_filters(self, filters):
        """Validate the filters.

        Parameters
        ----------
        filters: dict-like
            It is a dictionary in which the key is the name of a
            criterion, and the value is the filter condition.

        Returns
        -------
        (criteria, filters): tuple of two elements.
            The tuple contains two iterables:

            1. The first is the list of criteria.
            2. The second is the filters.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_mask(self, matrix, criteria, criteria_to_use, criteria_filters):
        raise NotImplementedError()

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, criteria, alternatives, **kwargs):
        # determine which criteria defined in the filter are in the DM
        criteria_to_use, criteria_filters = [], []
        for crit, flt in zip(self._criteria, self._filters):
            if crit not in criteria and not self._ignore_missing_criteria:
                raise ValueError(f"Missing criteria: {crit}")
            elif crit in criteria:
                criteria_to_use.append(crit)
                criteria_filters.append(flt)

        if criteria_to_use:

            mask = self._make_mask(
                matrix=matrix,
                criteria=criteria,
                criteria_to_use=criteria_to_use,
                criteria_filters=criteria_filters,
            )

            filtered_matrix = matrix[mask]
            filtered_alternatives = alternatives[mask]

        else:
            filtered_matrix = matrix
            filtered_alternatives = alternatives

        kwargs.update(
            matrix=filtered_matrix,
            criteria=criteria,
            alternatives=filtered_alternatives,
            dtypes=None,
        )
        return kwargs


# =============================================================================
# GENERIC FILTER
# =============================================================================


@doc_inherit(SKCByCriteriaFilterABC, warn_class=False)
class Filter(SKCByCriteriaFilterABC):
    """Function based filter.

    This class accepts as a filter any arbitrary function that receives as a
    parameter a as a parameter a criterion and returns a mask of the same size
    as the number of the number of alternatives in the decision matrix.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

         >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.Filter({
        ...     "ROE": lambda e: e > 1,
        ...     "RI": lambda e: e >= 28,
        ... })
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        PE          7          5        35
        AA          5          6        28
        FN          5          8        30
        [3 Alternatives x 3 Criteria]

    """

    def _coerce_filters(self, filters):
        criteria, criteria_filters = [], []
        for filter_name, filter_value in filters.items():
            if not isinstance(filter_name, str):
                raise ValueError("All filter keys must be instance of 'str'")
            if not callable(filter_value):
                raise ValueError("All filter values must be callable")
            criteria.append(filter_name)
            criteria_filters.append(filter_value)
        return tuple(criteria), tuple(criteria_filters)

    def _make_mask(self, matrix, criteria, criteria_to_use, criteria_filters):
        mask_list = []
        for crit_name, crit_filter in zip(criteria_to_use, criteria_filters):
            crit_idx = np.in1d(criteria, crit_name, assume_unique=False)
            crit_array = matrix[:, crit_idx].flatten()
            crit_mask = np.apply_along_axis(
                crit_filter, axis=0, arr=crit_array
            )
            mask_list.append(crit_mask)

        mask = np.all(np.column_stack(mask_list), axis=1)

        return mask


# =============================================================================
# ARITHMETIC FILTER
# =============================================================================


@doc_inherit(SKCByCriteriaFilterABC, warn_class=False)
class SKCArithmeticFilterABC(SKCByCriteriaFilterABC):
    """Provide a common behavior to make filters based on the same comparator.

    This abstract class require to redefine ``_filter`` method, and this will
    apply to each criteria separately.

    This class is designed to implement in general arithmetic comparisons of
    "==", "!=", ">", ">=", "<", "<=" taking advantage of the functions
    provided by numpy (e.g. ``np.greater_equal()``).

    Notes
    -----
    The filter implemented with this class are slightly faster than
    function-based filters.

    """

    _skcriteria_abstract_class = True

    @abc.abstractmethod
    def _filter(self, arr, cond):
        raise NotImplementedError()

    def _coerce_filters(self, filters):
        criteria, criteria_filters = [], []
        for filter_name, filter_value in filters.items():
            if not isinstance(filter_name, str):
                raise ValueError("All filter keys must be instance of 'str'")
            if not isinstance(filter_value, (int, float, complex, np.number)):
                raise ValueError(
                    "All filter values must be some kind of number"
                )
            criteria.append(filter_name)
            criteria_filters.append(filter_value)
        return tuple(criteria), tuple(criteria_filters)

    def _make_mask(self, matrix, criteria, criteria_to_use, criteria_filters):

        idxs = np.in1d(criteria, criteria_to_use)
        matrix = matrix[:, idxs]
        mask = np.all(self._filter(matrix, criteria_filters), axis=1)

        return mask


@doc_inherit(SKCArithmeticFilterABC, warn_class=False)
class FilterGT(SKCArithmeticFilterABC):
    """Keeps the alternatives for which the criteria value are greater than a \
    value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterGT({"ROE": 1, "RI": 27})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        PE          7          5        35
        AA          5          6        28
        FN          5          8        30
        [3 Alternatives x 3 Criteria]

    """

    _filter = np.greater


@doc_inherit(SKCArithmeticFilterABC, warn_class=False)
class FilterGE(SKCArithmeticFilterABC):
    """Keeps the alternatives for which the criteria value are greater or \
    equal than a value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterGE({"ROE": 1, "RI": 27})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        PE          7          5        35
        AA          5          6        28
        MM          1          7        30
        FN          5          8        30
        [4 Alternatives x 3 Criteria]

    """

    _filter = np.greater_equal


@doc_inherit(SKCArithmeticFilterABC, warn_class=False)
class FilterLT(SKCArithmeticFilterABC):
    """Keeps the alternatives for which the criteria value are less than a \
    value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterLT({"RI": 28})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        JN          5          4        26
        [1 Alternatives x 3 Criteria]

    """

    _filter = np.less


@doc_inherit(SKCArithmeticFilterABC, warn_class=False)
class FilterLE(SKCArithmeticFilterABC):
    """Keeps the alternatives for which the criteria value are less or equal \
    than a value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterLE({"RI": 28})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        JN          5          4        26
        AA          5          6        28
        [2 Alternatives x 3 Criteria]

    """

    _filter = np.less_equal


@doc_inherit(SKCArithmeticFilterABC, warn_class=False)
class FilterEQ(SKCArithmeticFilterABC):
    """Keeps the alternatives for which the criteria value are equal than a \
    value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterEQ({"CAP": 7, "RI": 30})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        MM          1          7        30
        [1 Alternatives x 3 Criteria]

    """

    _filter = np.equal


@doc_inherit(SKCArithmeticFilterABC, warn_class=False)
class FilterNE(SKCArithmeticFilterABC):
    """Keeps the alternatives for which the criteria value are not equal than \
    a value.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterNE({"CAP": 7, "RI": 30})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        PE          7          5        35
        JN          5          4        26
        AA          5          6        28
        [3 Alternatives x 3 Criteria]

    """

    _filter = np.not_equal


# =============================================================================
# SET FILTERS
# =============================================================================


@doc_inherit(SKCByCriteriaFilterABC, warn_class=False)
class SKCSetFilterABC(SKCByCriteriaFilterABC):
    """Provide a common behavior to make filters based on set operations.

    This abstract class require to redefine ``_set_filter`` method, and this
    will apply to each criteria separately.

    This class is designed to implement in general set comparison like
    "inclusion" and "exclusion".

    """

    _skcriteria_abstract_class = True

    @abc.abstractmethod
    def _set_filter(self, arr, cond):
        raise NotImplementedError()

    def _coerce_filters(self, filters):
        criteria, criteria_filters = [], []
        for filter_name, filter_value in filters.items():
            if not isinstance(filter_name, str):
                raise ValueError("All filter keys must be instance of 'str'")

            if not (
                isinstance(filter_value, Collection) and len(filter_value)
            ):
                raise ValueError(
                    "All filter values must be iterable with length > 1"
                )
            criteria.append(filter_name)
            criteria_filters.append(np.asarray(filter_value))
        return criteria, criteria_filters

    def _make_mask(self, matrix, criteria, criteria_to_use, criteria_filters):
        mask_list = []
        for fname, fset in zip(criteria_to_use, criteria_filters):
            crit_idx = np.in1d(criteria, fname, assume_unique=False)
            crit_array = matrix[:, crit_idx].flatten()
            crit_mask = self._set_filter(crit_array, fset)
            mask_list.append(crit_mask)

        mask = np.all(np.column_stack(mask_list), axis=1)

        return mask


@doc_inherit(SKCSetFilterABC, warn_class=False)
class FilterIn(SKCSetFilterABC):
    """Keeps the alternatives for which the criteria value are included in a \
    set of values.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterIn({"ROE": [7, 1], "RI": [30, 35]})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        PE          7          5        35
        MM          1          7        30
        [2 Alternatives x 3 Criteria]

    """

    def _set_filter(self, arr, cond):
        return np.isin(arr, cond)


@doc_inherit(SKCSetFilterABC, warn_class=False)
class FilterNotIn(SKCSetFilterABC):
    """Keeps the alternatives for which the criteria value are not included \
    in a set of values.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterNotIn({"ROE": [7, 1], "RI": [30, 35]})
        >>> tfm.transform(dm)
           ROE[▲ 2.0] CAP[▲ 4.0] RI[▼ 1.0]
        JN          5          4        26
        AA          5          6        28
        [2 Alternatives x 3 Criteria]

    """

    def _set_filter(self, arr, cond):
        return np.isin(arr, cond, invert=True)


# =============================================================================
# DOMINANCE
# =============================================================================


class FilterNonDominated(SKCTransformerABC):
    """Keeps the non dominated or non strictly-dominated alternatives.

    In order to evaluate the dominance of an alternative *a0* over an
    alternative *a1*, the algorithm evaluates that *a0* is better in at
    least one criterion and that *a1* is not better in any criterion than
    *a0*. In the case that ``strict = True`` it also evaluates that there
    are no equal criteria.

    Parameters
    ----------
    strict: bool, default ``False``
        If ``True``, strictly dominated alternatives are removed, otherwise all
        dominated alternatives are removed.

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import filters

        >>> dm = skc.mkdm(
        ...     matrix=[
        ...         [7, 5, 35],
        ...         [5, 4, 26],
        ...         [5, 6, 28],
        ...         [1, 7, 30],
        ...         [5, 8, 30]
        ...     ],
        ...     objectives=[max, max, min],
        ...     alternatives=["PE", "JN", "AA", "MM", "FN"],
        ...     criteria=["ROE", "CAP", "RI"],
        ... )

        >>> tfm = filters.FilterNonDominated(strict=False)
        >>> tfm.transform(dm)
           ROE[▲ 1.0] CAP[▲ 1.0] RI[▼ 1.0]
        PE          7          5        35
        JN          5          4        26
        AA          5          6        28
        FN          5          8        30
        [4 Alternatives x 3 Criteria]

    """

    _skcriteria_parameters = ["strict"]

    def __init__(self, *, strict=False):
        self._strict = bool(strict)

    @property
    def strict(self):
        """If the filter must remove the dominated or strictly-dominated \
        alternatives."""
        return self._strict

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, alternatives, dominated_mask, **kwargs):

        filtered_matrix = matrix[~dominated_mask]
        filtered_alternatives = alternatives[~dominated_mask]

        kwargs.update(
            matrix=filtered_matrix,
            alternatives=filtered_alternatives,
        )
        return kwargs

    @doc_inherit(SKCTransformerABC.transform)
    def transform(self, dm):

        data = dm.to_dict()
        dominated_mask = dm.dominance.dominated(strict=self._strict).to_numpy()

        transformed_data = self._transform_data(
            dominated_mask=dominated_mask, **data
        )

        transformed_dm = DecisionMatrix.from_mcda_data(**transformed_data)

        return transformed_dm
