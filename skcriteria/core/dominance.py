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

    _default_kind = "dominance"

    def __init__(self, dm):
        self._dm = dm

    @property
    @functools.lru_cache(maxsize=None)
    def _dominance_cache(self):
        """Cache of dominance.

        Compute the dominance is an O(n_C_2) algorithm, so lets use a cache.

        """
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
        """Return the entry of the cache.

        The input returned is the one that relates the alternatives a0 and a1.
        Since the cache can store the entry with the key (a0, a1) or (a1, a0),
        a second value is returned that is True if it was necessary to invert
        the alternatives.

        """
        key = a0, a1
        cache = self._dominance_cache
        entry, key_reverted = (
            (cache[key], False) if key in cache else (cache[key[::-1]], True)
        )
        return entry, key_reverted

    # FRAME ALT VS ALT ========================================================

    def _create_frame(self, compute_cell, iname, cname):
        """Create a data frame comparing two alternatives.

        The value of each cell is calculated with the "compute_cell"
        function.

        """
        alternatives = self._dm.alternatives
        rows = []
        for a0 in alternatives:
            row = OrderedDict()
            for a1 in alternatives:
                row[a1] = compute_cell(a0, a1)
            rows.append(row)

        df = pd.DataFrame(rows, index=alternatives)

        df.index.name = iname
        df.columns.name = cname

        return df

    def bt(self):
        """Compare on how many criteria one alternative is better than another.

        *bt* = better-than.

        Returns
        -------
        pandas.DataFrame:
            Where the value of each cell identifies on how many criteria the
            row alternative is better than the column alternative.

        """

        def compute_cell(a0, a1):
            if a0 == a1:
                return 0
            centry, ckreverted = self._cache_read(a0, a1)
            return centry.aDb if not ckreverted else centry.bDa

        return self._create_frame(
            compute_cell, iname="Better than", cname="Worse than"
        )

    def eq(self):
        """Compare on how many criteria two alternatives are equal.

        Returns
        -------
        pandas.DataFrame:
            Where the value of each cell identifies how many criteria the row
            and column alternatives are equal.

        """
        alternatives_len = len(self._dm.alternatives)

        def compute_cell(a0, a1):
            if a0 == a1:
                return alternatives_len
            centry, _ = self._cache_read(a0, a1)
            return centry.eq

        return self._create_frame(
            compute_cell, iname="Equals to", cname="Equals to"
        )

    def dominance(self, *, strict=False):
        """Compare if one alternative dominates or strictly dominates another \
        alternative.

        In order to evaluate the dominance of an alternative *a0* over an
        alternative *a1*, the algorithm evaluates that *a0* is better in at
        least one criterion and that *a1* is not better in any criterion than
        *a0*. In the case that ``strict = True`` it also evaluates that there
        are no equal criteria.

        Parameters
        ----------
        strict: bool, default ``False``
            If True, strict dominance is evaluated.

        Returns
        -------
        pandas.DataFrame:
            Where the value of each cell is True if the row alternative
            dominates the column alternative.

        """

        def compute_cell(a0, a1):
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

        iname, cname = (
            ("Strict dominators", "Strictly dominated")
            if strict
            else ("Dominators", "Dominated")
        )

        dom = self._create_frame(compute_cell, iname=iname, cname=cname)

        return dom

    # COMPARISONS =============================================================

    def compare(self, a0, a1):
        """Compare two alternatives.

        It creates a summary data frame containing the comparison of the two
        alternatives on a per-criteria basis, indicating which of the two is
        the best value, or if they are equal. In addition, it presents a
        "Performance" column with the count for each case.

        Parameters
        ----------
        a0, a1: str
            Names of the alternatives to compare.

        Returns
        -------
        pandas.DataFrame:
            Comparison of the two alternatives by criteria.

        """
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

    # The dominated ===========================================================

    def dominated(self, *, strict=False):
        """Which alternative is dominated or strictly dominated by at least \
        one other alternative.

        Parameters
        ----------
        strict: bool, default ``False``
            If True, strict dominance is evaluated.

        Returns
        -------
        pandas.Series:
            Where the index indicates the name of the alternative, and if the
            value is is True, it indicates that this alternative is dominated
            by at least one other alternative.

        """
        dom = self.dominance(strict=strict).any()
        dom.name = dom.index.name
        dom.index.name = "Alternatives"
        return dom

    @functools.lru_cache(maxsize=None)
    def dominators_of(self, a, *, strict=False):
        """Array of alternatives that dominate or strictly-dominate the \
        alternative provided by parameters.

        Parameters
        ----------
        a : str
            On what alternative to look for the dominators.
        strict: bool, default ``False``
            If True, strict dominance is evaluated.

        Returns
        -------
        numpy.ndarray:
            List of alternatives that dominate ``a``.

        """
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

    def has_loops(self, *, strict=False):
        """Retorna True si la matriz contiene loops de dominacia.

        A loop is defined as if there are alternatives `a0`, `a1` and 'a2' such
        that "a0 ≻ a1 ≻ a2 ≻ a0" if ``strict=True``, or "a0 ≽ a1 ≽ a2 ≽ a0"
        if ``strict=False``

        Parameters
        ----------
        strict: bool, default ``False``
            If True, strict dominance is evaluated.

        Returns
        -------
        bool:
            If True a loop exists.

        Notes
        -----
        If the result of this method is True, the ``dominators_of()`` method
        raises a ``RecursionError`` for at least one alternative.

        """
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
