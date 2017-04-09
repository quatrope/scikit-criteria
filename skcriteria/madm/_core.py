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
# IMPORTS
# =============================================================================

from __future__ import unicode_literals

import sys
import operator
import uuid
from collections import Mapping

import six

import numpy as np

from tabulate import tabulate

from .. import util, norm
from .._oop import Data, BaseSolver


# =============================================================================
# CONSTANTS
# =============================================================================

CRITERIA_AS_ATR = {
    util.MIN: "min",
    util.MAX: "max"
}

TABULATE_PARAMS = {
    "headers": "firstrow",
    "numalign": "center",
    "stralign": "center",
}


# =============================================================================
# EXTRA
# =============================================================================

class Extra(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    def __eq__(self, obj):
        if not isinstance(obj, Extra):
            return False
        if sorted(self._data.keys()) != sorted(obj._data.keys()):
            return False
        for k, v in self._data.items():
            ov = obj._data[k]
            if not isinstance(ov, type(v)):
                return False
            eq = util.iter_equal if isinstance(v, np.ndarray) else operator.eq
            if not eq(v, ov):
                return False
        return True

    def __ne__(self, obj):
        return not self == obj

    def __getitem__(self, k):
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        try:
            return self._data[k]
        except KeyError:
            msg = "'Extra' object has no attribute '{}'".format(k)
            raise AttributeError(msg)

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

    def to_str(self):
        return "Extra({})".format(", ".join(self._data))


# =============================================================================
# DECISION
# =============================================================================

class Decision(object):

    def __init__(self, decision_maker, data, kernel_, rank_, e_):
            self._decision_maker = decision_maker
            self._data = data
            self._kernel = kernel_
            self._rank = rank_
            self._e = Extra(e_)

    def _iter_rows(self):
        title = []
        if self._rank is not None:
            title.append("Rank")
        if self._kernel is not None:
            title.append("Kernel")
        for idx, row in enumerate(self._data._iter_rows()):
            if idx == 0:
                extra = title
            else:
                aidx = idx - 1
                extra = []
                if self._rank is not None:
                    extra.append(self._rank[aidx])
                if self._kernel is not None:
                    extra.append("  @" if idx - 1 in self._kernel else "")
            yield row + extra

    def __eq__(self, obj):
        return (
            isinstance(obj, Decision) and
            self._decision_maker == obj._decision_maker and
            self._data == obj._data and
            util.iter_equal(self._kernel, obj._kernel) and
            util.iter_equal(self._rank, obj._rank) and
            self._e == obj._e)

    def __ne__(self, obj):
        return not self == obj

    def __unicode__(self):
        return "{} - Solution:\n{}".format(
            repr(self._decision_maker)[1: -1], self.to_str())

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
        uid = "dec-" + str(uuid.uuid1())
        table = self.to_str(tablefmt="html")
        dm = repr(self._decision_maker)[1: -1]
        return "<div id='{}'><p><b>{} - Solution:</b></p>{}</div>".format(
            uid, dm, table)

    def to_str(self, **params):
        params.update({
            k: v for k, v in TABULATE_PARAMS.items() if k not in params})
        rows = self._iter_rows()
        return tabulate(rows, **params)

    def as_dict(self):
        data = {
            "data": self._data.as_dict(),
            "kernel_": self._kernel,
            "rank_": self._rank, "e_": self._e}
        dm = self._decision_maker.as_dict()
        data.update({"decision_maker": dm})
        return data

    @property
    def decision_maker(self):
        return self._decision_maker

    @property
    def data(self):
        return self._data

    @property
    def mtx(self):
        return self._data.mtx

    @property
    def criteria(self):
        return self._data.criteria

    @property
    def weights(self):
        return self._data.weights

    @property
    def kernel_(self):
        return self._kernel

    @property
    def rank_(self):
        return self._rank

    @property
    def e_(self):
        return self._e

    @property
    def best_alternative_(self):
        if self._rank is not None:
            return self._rank[0]

    @property
    def alpha_solution_(self):
        return self._rank is not None

    @property
    def beta_solution_(self):
        return self._kernel is not None

    @property
    def gamma_solution_(self):
        return self._rank is not None


# =============================================================================
# DECISION MAKER
# =============================================================================

class DecisionMaker(BaseSolver):

    def __init__(self, mnorm, wnorm):
        self._mnorm = mnorm if hasattr(mnorm, "__call__") else norm.get(mnorm)
        self._wnorm = wnorm if hasattr(wnorm, "__call__") else norm.get(wnorm)

    def as_dict(self):
        return {"mnorm": norm.nameof(self._mnorm),
                "wnorm": norm.nameof(self._wnorm)}

    def preprocess(self, data):
        ncriteria = util.criteriarr(data.criteria)
        nmtx = self._mnorm(data.mtx, axis=0)
        nweights = self._wnorm(data.weights) if data.weights is not None else 1
        return Data(mtx=nmtx, criteria=ncriteria, weights=nweights,
                    anames=data.anames, cnames=data.cnames)

    def make_result(self, data, kernel, rank, extra):
        decision = Decision(
            decision_maker=self, data=data,
            kernel_=kernel, rank_=rank, e_=extra)
        return decision

    @property
    def mnorm(self):
        return self._mnorm

    @property
    def wnorm(self):
        return self._wnorm
