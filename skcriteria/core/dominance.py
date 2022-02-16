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
import itertools as it
from collections import OrderedDict

import numpy as np

import pandas as pd

from ..utils import AccessorABC, rank

# =============================================================================
# DOMINANCE ACCESSOR
# =============================================================================


class DecisionMatrixDominanceAccessor(AccessorABC):
    """Calculate basic statistics of the decision matrix."""

    _DEFAULT_KIND = "dominance"

    def __init__(self, dm):
        self._dm = dm

    @property
    @functools.lru_cache(maxsize=None)
    def _dominance_cache(self):
        # Compute the dominance is an 0^2 algorithm, so lets use a cache
        dm = self._dm

        reverse = dm.minwhere

        dominance_cache, alts_numpy = {}, {}

        for a0, a1 in it.combinations(dm.alternatives, 2):
            for aname in (a0, a1):
                if aname not in alts_numpy:
                    alts_numpy[aname] = dm.alternatives[aname]

            dominance_cache[(a0, a1)] = rank.dominance(
                alts_numpy[a0], alts_numpy[a1], reverse=reverse
            )

        return dominance_cache

    def _cache_read(self, a0, a1):
        key = a0, a1
        cache = self._dominance_cache
        entry, key_reverted = (
            (cache[key], False) if key in cache else (cache[key[::-1]], True)
        )
        return entry, key_reverted

    def _create_frame(self, extract):
        alternatives = self._dm.alternatives
        rows = []
        for a0 in alternatives:
            row = OrderedDict()
            for a1 in alternatives:
                row[a1] = extract(a0, a1)
            rows.append(row)
        return pd.DataFrame(rows, index=alternatives)

    def bt(self):
        def extract(a0, a1):
            if a0 == a1:
                return 0
            centry, ckreverted = self._cache_read(a0, a1)
            return centry.aDb if not ckreverted else centry.bDa

        return self._create_frame(extract)

    def eq(self):
        alternatives_len = len(self._dm.alternatives)

        def extract(a0, a1):
            if a0 == a1:
                return alternatives_len
            centry, _ = self._cache_read(a0, a1)
            return centry.eq

        return self._create_frame(extract)

    def resume(self, a0, a1):

        # read the cache and extract the values
        centry, ckreverted = self._cache_read(a0, a1)
        performance_a0, performance_a1 = (
            (centry.aDb, centry.bDa)
            if not ckreverted
            else (centry.bDa, centry.aDb)
        )
        where_aDb, where_bDa = (
            (centry.aDb_where, centry.bDa_where)
            if not ckreverted
            else (centry.bDa_where, centry.aDb_where)
        )
        eq, eq_where = centry.eq, centry.eq_where

        criteria = self._dm.criteria

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
                pd.Series(eq_where, name=alt_index[2], index=crit_index),
            ]
        )

        df = df.assign(
            Performance=[performance_a0, performance_a1, eq],
        )

        return df

    def dominance(self, strict=False):
        def extract(a0, a1):
            if a0 == a1:
                return False
            centry, ckreverted = self._cache_read(a0, a1)
            performance_a0, performance_a1 = (
                (centry.aDb, centry.bDa)
                if not ckreverted
                else (centry.bDa, centry.aDb)
            )

            if strict and centry.eq:
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
