#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Dominance helper for the DecisionMatrix object."""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
from collections import OrderedDict

import numpy as np

import pandas as pd


# =============================================================================
# DOMINANCE ACCESSOR
# =============================================================================


class DecisionMatrixDominanceAccessor:
    """Calculate basic statistics of the decision matrix."""

    _DEFAULT_KIND = "dominance"

    def __init__(self, dm, dominance_cache):
        self._dm = dm
        self._dominance_cache = dominance_cache

    def __call__(self, kind=None, **kwargs):
        """Calculate basic statistics of the decision matrix.

        Parameters
        ----------


        """
        kind = self._DEFAULT_KIND if kind is None else kind

        if kind.startswith("_"):
            raise ValueError(f"invalid kind name '{kind}'")

        method = getattr(self, kind, None)
        if not callable(method):
            raise ValueError(f"Invalid kind name '{kind}'")

        return method(**kwargs)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        return f"{type(self).__name__}({self._dm!r})"

    def _create_frame(self, extract):
        alternatives, domdict = self._dm.alternatives, self._dominance_cache
        rows = []
        for a0 in alternatives:
            row = OrderedDict()
            for a1 in alternatives:
                row[a1] = extract(a0, a1, domdict)
            rows.append(row)
        return pd.DataFrame(rows, index=alternatives)

    def bt(self):
        def extract(a0, a1, domdict):
            if a0 == a1:
                return 0
            try:
                return domdict[(a0, a1)].aDb
            except KeyError:
                return domdict[(a1, a0)].bDa

        return self._create_frame(extract)

    def eq(self):
        alternatives_len = len(self._dm.alternatives)

        def extract(a0, a1, domdict):
            if a0 == a1:
                return alternatives_len
            try:
                return domdict[(a0, a1)].eq
            except KeyError:
                return domdict[(a1, a0)].eq

        return self._create_frame(extract)

    def resume(self, a0, a1):
        domdict = self._dominance_cache
        criteria = self._dm.criteria

        try:
            info = domdict[(a0, a1)]
            performance_a0, performance_a1 = info.aDb, info.bDa
            where_aDb, where_bDa = info.aDb_where, info.bDa_where
        except KeyError:
            info = domdict[(a1, a0)]
            performance_a1, performance_a0 = info.aDb, info.bDa
            where_bDa, where_aDb = info.aDb_where, info.bDa_where

        alt_index = pd.MultiIndex.from_tuples(
            [
                ("Alternatives", a0),
                ("Alternatives", a1),
                ("Equals", ""),
            ]
        )
        crit_index = pd.MultiIndex.from_product([["Criteria"], criteria])

        df = pd.DataFrame(
            [
                pd.Series(where_aDb, name=alt_index[0], index=crit_index),
                pd.Series(where_bDa, name=alt_index[1], index=crit_index),
                pd.Series(info.eq_where, name=alt_index[2], index=crit_index),
            ]
        )

        df = df.assign(
            Performance=[performance_a0, performance_a1, info.eq],
        )

        return df

    def dominance(self, strict=False):
        def extract(a0, a1, domdict):
            if a0 == a1:
                return False
            try:
                info = domdict[(a0, a1)]
                performance_a0, performance_a1 = info.aDb, info.bDa
            except KeyError:
                info = domdict[(a1, a0)]
                performance_a1, performance_a0 = info.aDb, info.bDa

            if strict and info.eq:
                return False

            return performance_a0 > 0 and performance_a1 == 0

        return self._create_frame(extract)

    def dominated(self, strict=False):
        return self.dominance(strict=strict).any()

    @functools.lru_cache(maxsize=None)
    def dominators_of(self, a, strict=False):

        dominance_a = self.dominance(strict=strict)[a]
        if ~dominance_a.any():
            return np.array([], dtype=str)

        dominators = dominance_a.index[dominance_a]
        for dominator in dominators:
            dominators_dominators = self.dominators_of(
                dominator, strict=strict
            )
            dominators = np.concatenate((dominators, dominators_dominators))
        return dominators

    def has_loops(self, strict=False):
        # lets put the dominated alternatives last so our while loop will
        # be shorter by extracting from the tail

        alternatives = list(self.dominated(strict=strict).sort_values().index)

        try:
            while alternatives:
                # dame la ultima alternativa (al final quedan las dominadas)
                alt = alternatives.pop()

                # ahora dame todas las alternatives las cuales dominan
                dominators = self.dominators_of(alt, strict=strict)

                # las alternativas dominadoras ya pasaron por "dominated_by"
                # por lo cual hay que sacarlas a todas de alternatives
                alternatives = [a for a in alternatives if a not in dominators]

        except RecursionError:
            return True
        return False
