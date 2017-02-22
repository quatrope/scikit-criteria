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

import abc
import operator
import uuid
from collections import Mapping

import six

import numpy as np

from tabulate import tabulate

from . import util, norm


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
# CLASSES
# =============================================================================

class Data(object):

    def __init__(self, mtx, criteria, weights=None, anames=None, cnames=None):
        self._mtx = np.asarray(mtx)
        self._criteria = util.criteriarr(criteria)
        self._weights = np.asarray(weights) if weights is not None else None
        util.validate_data(self._mtx, self._criteria, self._weights)

        self._anames = (
            anames if anames else
            ["A{}".format(idx) for idx in range(len(mtx))])
        if len(self._anames) != len(self._mtx):
            msg = "{} names given for {} alternatives".format(
                len(self._anames), len(self._mtx))
            raise util.DataValidationError(msg)

        self._cnames = (
            cnames if cnames else
            ["C{}".format(idx) for idx in range(len(criteria))])
        if len(self._cnames) != len(self._criteria):
            msg = "{} names for given {} criteria".format(
                len(self._cnames), len(self._criteria))
            raise util.DataValidationError(msg)

    def _iter_rows(self):
        direction = map(CRITERIA_AS_ATR.get, self._criteria)
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
            util.iter_equal(self._mtx, obj._mtx) and
            util.iter_equal(self._criteria, obj._criteria) and
            util.iter_equal(self._weights, obj._weights))

    def __ne__(self, obj):
        return not self == obj

    def __str__(self):
        return self.to_str()

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


class Extra(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    def __json_encode__(self):
        return {"data": self._data}

    def __json_decode__(self, data):
        self.__init__(data)

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

    def __repr__(self):
        return "Extra({})".format(", ".join(self._data))


class Decision(object):

    def __init__(self, decision_maker, mtx, criteria, weights,
                 anames, cnames, kernel_, rank_, e_,):
            self._decision_maker = decision_maker
            self._data = Data(mtx, criteria, weights,
                              anames=anames, cnames=cnames)
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
                    extra.append("  @" if idx in self._kernel else "")
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

    def __json_encode__(self):
        return self.as_dict()

    def __json_decode__(self, **attrs):
        decision_maker = attrs.pop("decision_maker")
        data = decision_maker.decision_from_dict(attrs)
        data.update({"decision_maker": decision_maker})

        data_instance = attrs.pop("data")
        data.update({"mtx": data_instance.mtx,
                     "criteria": data_instance.criteria,
                     "weights": data_instance.weights,
                     "anames": data_instance.anames,
                     "cnames": data_instance.cnames})
        self.__init__(**data)

    def __str__(self):
        return "{} - Solution:\n{}".format(
            repr(self._decision_maker)[1: -1], self.to_str())

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
            "data": self._data,
            "kernel_": self._kernel,
            "rank_": self._rank, "e_": self._e}
        data = self.decision_maker.decision_as_dict(data)
        data.update({"decision_maker": self._decision_maker})
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

@six.add_metaclass(abc.ABCMeta)
class DecisionMaker(object):

    def __init__(self, mnorm, wnorm):
        self._mnorm = mnorm if hasattr(mnorm, "__call__") else norm.get(mnorm)
        self._wnorm = wnorm if hasattr(wnorm, "__call__") else norm.get(wnorm)

    def __eq__(self, obj):
        return isinstance(obj, type(self)) and self.as_dict() == obj.as_dict()

    def __ne__(self, obj):
        return not self == obj

    def __json_encode__(self):
        # this is where jsontrick hangs on
        return self.as_dict()

    def __json_decode__(self, **attrs):
        # this is where jsontrick hangs on
        data = self.from_dict(attrs)
        self.__init__(**data)

    def __repr__(self):
        cls_name = type(self).__name__
        data = sorted(self.as_dict().items())
        data = ", ".join(
            "{}={}".format(k, v) for k, v in data)
        return "<{} ({})>".format(cls_name, data)

    def decision_as_dict(self, attrs):
        return attrs

    def decision_from_dict(self, data):
        return data

    def as_dict(self):
        try:
            return {"mnorm": norm.nameof(self._mnorm),
                    "wnorm": norm.nameof(self._wnorm)}
        except norm.FunctionNotRegisteredAsNormalizer as err:
            msg = ("All your normalization function must be registered with "
                   "'norm.register()' function. Invalid Function: {}")
            raise norm.FunctionNotRegisteredAsNormalizer(msg.format(err))

    def from_dict(self, data):
        return data

    def normalize(self, mtx, criteria, weights):
        ncriteria = util.criteriarr(criteria)
        nmtx = self._mnorm(mtx, axis=0)
        nweights = self._wnorm(weights) if weights is not None else 1
        return nmtx, ncriteria, nweights

    def make_decision(self, mtx, criteria, weights,
                      kernel, rank, extra, anames, cnames):
        decision = Decision(
            decision_maker=self,
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames,
            kernel_=kernel, rank_=rank, e_=extra)
        return decision

    def decide(self, data, criteria=None, weights=None):
        if isinstance(data, Data):
            if criteria or weights:
                msg = (
                    "If 'data' is instance of Data, 'criteria' and 'weights' "
                    "must be empty")
                raise ValueError(msg)
            anames, cnames = data.anames, data.cnames
            mtx, criteria, weights = data.mtx, data.criteria, data.weights
        else:
            if criteria is None:
                msg = (
                    "If 'data' is not instance of Data you must provide a "
                    "'criteria' array")
                raise ValueError(msg)
            anames, cnames, mtx = None, None, data
            util.validate_data(mtx, criteria, weights)
        nmtx, ncriteria, nweights = self.normalize(mtx, criteria, weights)
        kernel, rank, extra = self.solve(
            nmtx=nmtx, ncriteria=ncriteria, nweights=nweights)
        decision = self.make_decision(
            mtx=mtx, criteria=criteria, weights=weights,
            kernel=kernel, rank=rank, extra=extra,
            anames=anames, cnames=cnames)
        return decision

    @abc.abstractmethod
    def solve(self, nmtx, ncriteria, nweights):
        return NotImplemented

    @property
    def mnorm(self):
        return self._mnorm

    @property
    def wnorm(self):
        return self._wnorm
