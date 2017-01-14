#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    decision_maker_ = attr.ib()
    mtx_ = attr.ib()
    criteria_ = attr.ib()
    weights_ = attr.ib()
    efficients_ = attr.ib()
    rank_ = attr.ib()

    e = attr.ib(convert=_Extra)

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
            decision_maker_=self,
            mtx_=mtx, criteria_=criteria, weights_=weights,
            efficients_=efficients, rank_=rank, e=extra)
