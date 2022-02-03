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

from ..core import SKCTransformerABC
from ..utils import doc_inherit

# =============================================================================
# BASE CLASS
# =============================================================================


class SKCFilterABC(SKCTransformerABC):

    _skcriteria_parameters = frozenset(["criteria_filters"])
    _skcriteria_abstract_class = True

    def __init__(self, criteria_filters):
        if not len(criteria_filters):
            raise ValueError("Must provide at least one filter")

        criteria_filters = dict(criteria_filters)
        self._validate_filters(criteria_filters)
        self._criteria_filters = criteria_filters

    @property
    def criteria_filters(self):
        return dict(self._criteria_filters)

    @abc.abstractmethod
    def _validate_filters(self, filters):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_mask(self, matrix, criteria):
        raise NotImplementedError()

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, criteria, alternatives, **kwargs):
        criteria_not_found = set(self._criteria_filters).difference(criteria)
        if criteria_not_found:
            raise ValueError(f"Missing criteria: {criteria_not_found}")

        mask = self._make_mask(matrix, criteria)

        filtered_matrix = matrix[mask]
        filtered_alternatives = alternatives[mask]

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


class Filter(SKCFilterABC):
    def _validate_filters(self, filters):
        for filter_name, filter_value in filters.items():
            if not isinstance(filter_name, str):
                raise ValueError("All filter keys must be instance of 'str'")
            if not callable(filter_value):
                raise ValueError("All filter values must be callable")

    def _make_mask(self, matrix, criteria):
        mask_list = []
        for fname, fvalue in self._criteria_filters.items():
            crit_idx = np.in1d(criteria, fname, assume_unique=False)
            crit_array = matrix[:, crit_idx].flatten()
            crit_mask = np.apply_along_axis(fvalue, axis=0, arr=crit_array)
            mask_list.append(crit_mask)

        mask = np.all(np.column_stack(mask_list), axis=1)

        return mask


# =============================================================================
# ARITHMETIC FILTER
# =============================================================================


class SKCArithmeticFilterABC(SKCFilterABC):
    _skcriteria_abstract_class = True

    @abc.abstractmethod
    def _filter(self, arr, cond):
        raise NotImplementedError()

    def _validate_filters(self, filters):
        for filter_name, filter_value in filters.items():
            if not isinstance(filter_name, str):
                raise ValueError("All filter keys must be instance of 'str'")
            if not isinstance(filter_value, (int, float, complex, np.number)):
                raise ValueError(
                    "All filter values must be some kind of number"
                )

    def _make_mask(self, matrix, criteria):
        filter_names, filter_values = [], []

        for fname, fvalue in self._criteria_filters.items():
            filter_names.append(fname)
            filter_values.append(fvalue)

        idxs = np.in1d(criteria, filter_names)
        matrix = matrix[:, idxs]
        mask = np.all(self._filter(matrix, filter_values), axis=1)

        return mask


class FilterGT(SKCArithmeticFilterABC):

    _filter = np.greater


class FilterGE(SKCArithmeticFilterABC):

    _filter = np.greater_equal


class FilterLT(SKCArithmeticFilterABC):

    _filter = np.less


class FilterLE(SKCArithmeticFilterABC):

    _filter = np.less_equal


class FilterEQ(SKCArithmeticFilterABC):

    _filter = np.equal


class FilterNE(SKCArithmeticFilterABC):

    _filter = np.not_equal


# =============================================================================
# SET FILT
# =============================================================================


class SKCSetFilterABC(SKCFilterABC):
    _skcriteria_abstract_class = True

    @abc.abstractmethod
    def _set_filter(self, arr, cond):
        raise NotImplementedError()

    def _validate_filters(self, filters):
        for filter_name, filter_value in filters.items():
            if not isinstance(filter_name, str):
                raise ValueError("All filter keys must be instance of 'str'")

            if not (
                isinstance(filter_value, Collection) and len(filter_value)
            ):
                raise ValueError(
                    "All filter values must be iterable with length > 1"
                )

    def _make_mask(self, matrix, criteria):
        mask_list = []
        for fname, fset in self._criteria_filters.items():
            crit_idx = np.in1d(criteria, fname, assume_unique=False)
            crit_array = matrix[:, crit_idx].flatten()
            crit_mask = self._set_filter(crit_array, fset)
            mask_list.append(crit_mask)

        mask = np.all(np.column_stack(mask_list), axis=1)

        return mask


class FilterIn(SKCSetFilterABC):
    def _set_filter(self, arr, cond):
        return np.isin(arr, cond)


class FilterNotIn(SKCSetFilterABC):
    def _set_filter(self, arr, cond):
        return np.isin(arr, cond, invert=True)
