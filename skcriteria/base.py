#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# =============================================================================
# DOCS
# =============================================================================

"""Module containing the basic functionality
for the data representation used inside Scikit-Criteria.

"""

__all__ = ['Data']


# =============================================================================
# IMPORTS
# =============================================================================

import abc
from collections.abc import Mapping

import numpy as np

from tabulate import tabulate

from .utils.doc_inherit import InheritableDocstrings
from .utils.acc_property import AccessorProperty
from .validate import (CRITERIA_STR,
                       DataValidationError,
                       validate_data, iter_equal)
from .plot import DataPlotMethods


# =============================================================================
# CONSTANTS
# =============================================================================

TABULATE_PARAMS = {
    "headers": "firstrow",
    "numalign": "center",
    "stralign": "center",
}


# =============================================================================
# DATA
# =============================================================================

class MetaData(Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, n):
        try:
            return self._data[n]
        except KeyError:
            raise AttributeError(n)

    def __dir__(self):
        return list(self._data)

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    def to_str(self):
        return "MetaData(" + ", ".join(self) + ")"


class Data(object):
    """Multi-Criteria data representation.

    This make easy to manipulate:

    - The matrix of alternatives. (``mtx``)
    - The array with the sense of optimality of every criteria (``criteria``).
    - Optional weights of the criteria (``weights``)
    - Optional names of the alternatives (``anames``) and the
      criteria (``cnames``)
    - Optional metadata (``meta``)

    """

    def __init__(self, mtx, criteria,
                 weights=None, anames=None, cnames=None, meta=None):

        # validate and store all data
        self._mtx, self._criteria, self._weights = validate_data(
            mtx, criteria, weights)

        # validate alternative names
        self._anames = (
            anames if anames is not None else
            ["A{}".format(idx) for idx in range(len(mtx))])
        if len(self._anames) != len(self._mtx):
            msg = "{} names given for {} alternatives".format(
                len(self._anames), len(self._mtx))
            raise DataValidationError(msg)

        # validate criteria names
        self._cnames = (
            cnames if cnames is not None else
            ["C{}".format(idx) for idx in range(len(criteria))])
        if len(self._cnames) != len(self._criteria):
            msg = "{} names for given {} criteria".format(
                len(self._cnames), len(self._criteria))
            raise DataValidationError(msg)

        self._meta = MetaData(meta or {})

    def _iter_rows(self):
        direction = map(CRITERIA_STR.get, self._criteria)
        title = ["ALT./CRIT."]
        if self._weights is None:
            cstr = zip(self._cnames, direction)
            criteria = ["{} ({})".format(n, c) for n, c in cstr]
        else:
            cstr = zip(self._cnames, direction, self._weights)
            criteria = ["{} ({}) W.{}".format(n, c, w) for n, c, w in cstr]
        yield title + criteria

        for an, row in zip(self._anames, self._mtx):
            yield [an] + list(row)

    def __eq__(self, obj):
        return (
            isinstance(obj, Data) and self._meta == obj.meta and
            iter_equal(self._mtx, obj._mtx) and
            iter_equal(self._criteria, obj._criteria) and
            iter_equal(self._weights, obj._weights))

    def __ne__(self, obj):
        return not self == obj

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        return self.to_str(tablefmt="html")

    def to_str(self, **params):
        """String representation of the Data object.

        Parameters
        ----------

        kwargs :
            Parameters to configure
            `tabulate <https://bitbucket.org/astanin/python-tabulate>`_

        Returns
        -------

        str :
            String representation of the Data object.

        """
        params.update({
            k: v for k, v in TABULATE_PARAMS.items() if k not in params})
        rows = self._iter_rows()
        return tabulate(rows, **params)

    def raw(self):
        """Return a (mtx, criteria, weights, anames, cnames) tuple"""
        return self.mtx, self.criteria, self.weights, self.anames, self.cnames

    @property
    def anames(self):
        """Names of the alternatives as tuple of string."""
        return tuple(self._anames)

    @property
    def cnames(self):
        """Names of the criteria as tuple of string."""
        return tuple(self._cnames)

    @property
    def mtx(self):
        """Alternative matrix as 2d numpy.ndarray."""
        return self._mtx.copy()

    @property
    def criteria(self):
        """Sense of optimality of every criteria"""
        return self._criteria.copy()

    @property
    def weights(self):
        """Relative importance of the criteria or None if all the same"""
        return None if self._weights is None else self._weights.copy()

    @property
    def meta(self):
        """Dict-like metadata"""
        return self._meta

    # ----------------------------------------------------------------------
    # Add plotting methods to DataFrame

    plot = AccessorProperty(DataPlotMethods, DataPlotMethods)


# =============================================================================
# DECISION MAKER
# =============================================================================

class BaseSolverMeta(abc.ABCMeta, InheritableDocstrings):
    pass


class BaseSolver(metaclass=BaseSolverMeta):

    def __init__(self, mnorm, wnorm):
        from . import norm

        self._mnorm = norm.get(mnorm, mnorm)
        self._wnorm = norm.get(wnorm, wnorm)
        if not hasattr(self._mnorm, "__call__"):
            msg = "'mnorm' must be a callable or a string in {}. Found {}"
            raise TypeError(msg.format(norm.NORMALIZERS.keys(), mnorm))
        if not hasattr(self._wnorm, "__call__"):
            msg = "'wnorm' must be a callable or a string in {}. Found {}"
            raise TypeError(msg.format(norm.NORMALIZERS.keys(), wnorm))

    def __eq__(self, obj):
        return isinstance(obj, type(self)) and self.as_dict() == obj.as_dict()

    def __ne__(self, obj):
        return not self == obj

    def __str__(self):
        cls_name = type(self).__name__
        data = sorted(self.as_dict().items())
        data = ", ".join(
            "{}={}".format(k, v) for k, v in data)
        return "<{} ({})>".format(cls_name, data)

    def __repr__(self):
        return str(self)

    def as_dict(self):
        """Create a simply :py:class:`dict` representation of the object.

        Notes
        -----

        ``x.as_dict != dict(x)``

        """
        return {"mnorm": self._mnorm.__name__,
                "wnorm": self._wnorm.__name__}

    def preprocess(self, data):
        """Normalize the alternative matrix and weight vector.

        Creates a new instance of data by aplying the normalization function
        to the alternative matrix and the weights vector containded inside
        the given data.

        Parameters
        ----------

        data : :py:class:`skcriteria.Data`
            A data to be Preprocessed

        Returns
        -------

        :py:class:`skcriteria.Data`
            A new instance of data with the ``mtx`` attributes normalized
            with ``mnorm`` and ``weights`` normalized with wnorm. ``anames``
            and ``cnames`` are preseved

        """
        nmtx = self._mnorm(data.mtx, criteria=data.criteria, axis=0)
        nweights = (
            self._wnorm(data.weights, criteria=data.criteria)
            if data.weights is not None else
            np.ones(data.criteria.shape))
        return Data(mtx=nmtx, criteria=data.criteria, weights=nweights,
                    anames=data.anames, cnames=data.cnames)

    def decide(self, data, criteria=None, weights=None, **kwargs):
        """Execute the Solver over the given data.

        Parameters
        ----------

        data : :py:class:`skcriteria.Data` or array_like
            :py:class:`skcriteria.Data` instance; or a alternative matrix
            (2d array_like) `n` rows and `m` columns, where n is the number of
            alternatives and `m` is the number of criteria.
        criteria : None or array_like, optional
            If data is :py:class:`skcriteria.Data` must be ``None``. Otherwise
            must be a 1d array_like with `m` elements (number of criteria);
            only the values ``-1`` (for minimization) and ``1`` (maximization)
            are allowed.
        weights : None or array_like, optional
            - If data is :py:class:`skcriteria.Data` must be ``None``.
            - If data is 2d array_like and weights are ``None`` all the
              criteria has the same weight.
            - If data is 2d array_like and weights are 1d array_like with `m`
              elements (number of criteria); the i-nth element represent the
              importance of the i-nth criteria.
        kwargs : optional
            keywords arguments for the solve method

        Returns
        -------

        :py:class:`object`
            Check the documentation of ``make_result()``

        """
        if isinstance(data, Data):
            if (criteria, weights) != (None, None):
                raise ValueError("If 'data' is instance of Data, 'criteria' "
                                 "and 'weights' must be None")
        else:
            if criteria is None:
                raise ValueError("If 'data' is not instance of Data you must "
                                 "provide a 'criteria' array")
            data = Data(data, criteria, weights)
        pdata = self.preprocess(data)
        result = self.solve(pdata, **kwargs)
        return self.make_result(data, *result)

    @abc.abstractmethod
    def solve(self, pdata):
        """Execute the multi-criteria method.

        Parameters
        ----------

        data : :py:class:`skcriteria.Data`
            Preprocessed Data.

        Returns
        -------

        :py:class:`object`
            object or tuple of objects with the raw result data.

        """
        return NotImplemented

    @abc.abstractmethod
    def make_result(self, rdata):
        return NotImplemented

    @property
    def mnorm(self):
        """Normalization function for the alternative matrix."""
        return self._mnorm

    @property
    def wnorm(self):
        """Normalization function for the weights vector."""
        return self._wnorm
