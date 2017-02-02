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
from collections import Mapping

import six

import attr

import numpy as np

from . import util


# =============================================================================
# RESULT Structure
# =============================================================================

class _Extra(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, k):
        return self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            msg = "'_Extra' object has no attribute '{}'".format(k)
            raise AttributeError(msg)

    def __repr__(self):
        return "_Extra({})".format(", ".join(self._data))


@attr.s(frozen=True, repr=False, cmp=False)
class _Decision(object):
    decision_maker = attr.ib()
    mtx = attr.ib()
    criteria = attr.ib()
    weights = attr.ib()
    efficients_ = attr.ib()
    rank_ = attr.ib()

    e_ = attr.ib(convert=_Extra)

    def __repr__(self):
        decision_maker = type(self.decision_maker_).__name__
        return "<Decision of '{}'{}>".format(decision_maker, self.mtx_.shape)

    @property
    def best_alternative_(self):
        if self.rank_ is not None:
            return self._rank_[0]

    @property
    def alpha_solution_(self):
        return self.rank_ is not None

    @property
    def beta_solution_(self):
        return self.efficients_ is not None

    @property
    def gamma_solution_(self):
        return self.rank_ is not None


# =============================================================================
# DECISION MAKER
# =============================================================================

@six.add_metaclass(abc.ABCMeta)
class DecisionMaker(object):

    @abc.abstractmethod
    def solve(self, mtx, criteria, weights=None):
        return NotImplemented

    def decide(self, mtx, criteria, weights=None):
        mtx, criteria = np.asarray(mtx), util.criteriarr(criteria)
        weights = np.asarray(weights) if weights is not None else None
        efficients, rank, extra = self.solve(
            mtx=mtx, criteria=criteria, weights=weights)
        return _Decision(
            decision_maker=self,
            mtx=mtx, criteria=criteria, weights=weights,
            efficients_=efficients, rank_=rank, e_=extra)
