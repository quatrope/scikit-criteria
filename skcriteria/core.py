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
# FUTURE & DOCS
# =============================================================================

from __future__ import unicode_literals


__doc__ = """Module containing the basic functionality of scikit-criteria
including:

- Scikit-Criteria Data representation.
- Scikit-Criteria Criteria representation.
- Scikit-Criteria Data validation.

"""

__all__ = [
    'MIN', 'MAX',
    'criteriarr',
    'validate_data',
    'Data']


# =============================================================================
# IMPORTS
# =============================================================================

import sys
import abc

import six

import numpy as np

from tabulate import tabulate


# =============================================================================
# CONSTANTS
# =============================================================================

MIN = -1

MAX = 1

CRITERIA_STR = {
    MIN: "min",
    MAX: "max"
}

TABULATE_PARAMS = {
    "headers": "firstrow",
    "numalign": "center",
    "stralign": "center",
}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DataValidationError(ValueError):
    """Raised when some part of the multicriteria data (alternative matrix,
    criteria array or weights array) are not compatible with another part.

    """
    pass


# =============================================================================
# FUNCTIONS
# =============================================================================

def iter_equal(a, b):
    """Validate if two iterables are equals independently of their type"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.allclose(a, b, equal_nan=True)
    return a == b


def is_mtx(mtx, size=None):
    """Return True if mtx is two dimensional structure.

    If `size` is not None, must be a expected shape of the mtx

    """
    try:
        mtx = np.asarray(mtx)
        a, b = mtx.shape
        if size and (a, b) != size:
            return False
    except:
        return False
    return True


def criteriarr(criteria):
    """Validate if the iterable only contains MIN (-1) and MAX (1) criteria"""

    criteria = np.asarray(criteria)
    if np.setdiff1d(criteria, [MIN, MAX]):
        msg = "Criteria Array only accept '{}' or '{}' Values. Found {}"
        raise DataValidationError(msg.format(MAX, MIN, criteria))
    return criteria


def validate_data(mtx, criteria, weights=None):
    """Validate if the main components of the Data in scikit-criteria are
    compatible.

    The function tests:

    - The matrix (mtx) must be 2-dimensional.
    - The criteria array must be a criteria array (criteriarr function).
    - The number of criteria must be the same number of columns in mtx.
    - The weight array must be None or an iterable with the same length
      of the criteria.

    Returns
    -------

    - A mtx as 2d numpy.ndarray.
    - A criteria as numpy.ndarray.
    - A weights as numpy.ndarray or None (if weights is None).

    """
    mtx = np.asarray(mtx)
    if not is_mtx(mtx):
        msg = "'mtx' must be a 2 dimensional array"
        raise DataValidationError(msg)

    criteria = criteriarr(criteria)
    if len(criteria) != np.shape(mtx)[1]:
        msg = "{} senses of optimality given but mtx has {} criteria".format(
            len(criteria), np.shape(mtx)[1])
        raise DataValidationError(msg)

    weights = (np.asarray(weights) if weights is not None else None)
    if weights is not None:
        if len(weights) != len(criteria):
            msg = "{} weights given for {} criteria".format(
                len(weights), len(criteria))
            raise DataValidationError(msg)

    return mtx, criteria, weights


# =============================================================================
# DATA PROXY
# =============================================================================

class Data(object):

    def __init__(self, mtx, criteria, weights=None, anames=None, cnames=None):

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

        # create plot proxy
        from . import plot
        self._plot = plot.PlotProxy(self)

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
            isinstance(obj, Data) and
            iter_equal(self._mtx, obj._mtx) and
            iter_equal(self._criteria, obj._criteria) and
            iter_equal(self._weights, obj._weights))

    def __ne__(self, obj):
        return not self == obj

    def __unicode__(self):
        return self.to_str()

    def __bytes__(self):
        encoding = sys.getdefaultencoding()
        return self.__unicode__().encode(encoding, 'replace')

    def __str__(self):
        """Return a string representation for a particular Object

        Invoked by str(df) in both py2/py3.
        Yields Bytestring in Py2, Unicode String in py3.
        """
        if six.PY3:
            return self.__unicode__()
        return self.__bytes__()

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        return self.to_str(tablefmt="html")

    def to_str(self, **params):
        params.update({
            k: v for k, v in TABULATE_PARAMS.items() if k not in params})
        rows = self._iter_rows()
        return tabulate(rows, **params)

    @property
    def anames(self):
        return tuple(self._anames)

    @property
    def cnames(self):
        return tuple(self._cnames)

    @property
    def mtx(self):
        return self._mtx

    @property
    def criteria(self):
        return self._criteria

    @property
    def weights(self):
        return self._weights

    @property
    def plot(self):
        return self._plot


# =============================================================================
# DECISION MAKER
# =============================================================================

@six.add_metaclass(abc.ABCMeta)
class BaseSolver(object):

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
        return {"mnorm": self._mnorm.__name__,
                "wnorm": self._wnorm.__name__}

    def preprocess(self, data):
        nmtx = self._mnorm(data.mtx, criteria=data.criteria, axis=0)
        nweights = (
            self._wnorm(data.weights, criteria=data.criteria)
            if data.weights is not None else
            np.ones(data.criteria.shape))
        return Data(mtx=nmtx, criteria=data.criteria, weights=nweights,
                    anames=data.anames, cnames=data.cnames)

    def decide(self, data, criteria=None, weights=None):
        """foo"""
        if isinstance(data, Data):
            if criteria or weights:
                raise ValueError("If 'data' is instance of Data, 'criteria' "
                                 "and 'weights' must be empty")
        else:
            if criteria is None:
                raise ValueError("If 'data' is not instance of Data you must "
                                 "provide a 'criteria' array")
            data = Data(data, criteria, weights)
        pdata = self.preprocess(data)
        result = self.solve(pdata)
        return self.make_result(data, *result)

    @abc.abstractmethod
    def solve(self, pdata):
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
