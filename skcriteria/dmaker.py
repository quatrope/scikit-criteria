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
from collections import Mapping

import six

import numpy as np

from . import util, norm


# =============================================================================
# CLASSES
# =============================================================================

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
                 kernel_, rank_, e_):
            self._decision_maker = decision_maker
            self._mtx = mtx
            self._criteria = criteria
            self._weights = weights

            self._kernel = kernel_
            self._rank = rank_
            self._e = Extra(e_)

    def __repr__(self):
        decision_maker = type(self._decision_maker).__name__
        return "<Decision of '{}'{}>".format(decision_maker, self._mtx.shape)

    def __eq__(self, obj):
        return (
            isinstance(obj, Decision) and
            self._decision_maker == obj._decision_maker and
            util.iter_equal(self._mtx, obj._mtx) and
            util.iter_equal(self._criteria, obj._criteria) and
            util.iter_equal(self._weights, obj._weights) and
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
        self.__init__(**data)

    def as_dict(self):
        data = {
            "mtx": self._mtx,
            "criteria": self._criteria,
            "weights": self._weights,
            "kernel_": self._kernel,
            "rank_": self._rank, "e_": self._e}
        data = self.decision_maker.decision_as_dict(data)
        data.update({"decision_maker": self._decision_maker})
        return data

    @property
    def decision_maker(self):
        return self._decision_maker

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

    @abc.abstractmethod
    def solve(self, nmtx, ncriteria, nweights):
        return NotImplemented

    def normalize(self, mtx, criteria, weights):
        ncriteria = util.criteriarr(criteria)
        nmtx = self._mnorm(mtx, axis=0)
        nweights = self._wnorm(weights) if weights is not None else 1
        return nmtx, ncriteria, nweights

    def make_decision(self, mtx, criteria, weights, kernel, rank, extra):
        decision = Decision(
            decision_maker=self,
            mtx=mtx, criteria=criteria, weights=weights,
            kernel_=kernel, rank_=rank, e_=extra)
        return decision

    def decide(self, mtx, criteria, weights=None):
        nmtx, ncriteria, nweights = self.normalize(mtx, criteria, weights)
        kernel, rank, extra = self.solve(
            nmtx=nmtx, ncriteria=ncriteria, nweights=nweights)
        decision = self.make_decision(
            mtx=mtx, criteria=criteria, weights=weights,
            kernel=kernel, rank=rank, extra=extra)
        return decision

    @property
    def mnorm(self):
        return self._mnorm

    @property
    def wnorm(self):
        return self._wnorm
