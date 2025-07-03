#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tie breaker decision maker for eliminating ties in rankings."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import defaultdict

import numpy as np

import pandas as pd


from .agg import RankResult
from .core import SKCMethodABC
from .utils import doc_inherit


# =============================================================================
# CLASS
# =============================================================================


class TieBreaker(SKCMethodABC):
    """Decision maker that breaks ties in rankings using a secondary decision maker.

    This class takes a primary decision maker that may produce tied rankings
    and uses a secondary decision maker to break those ties. If the secondary
    decision maker also produces ties and force=True, it uses the untied_rank_
    property to ensure a complete ranking without ties.

    Parameters
    ----------
    dmaker : decision maker
        Primary decision maker that implements the `evaluate()` method.
        This decision maker may produce rankings with ties.

    untier : decision maker
        Secondary decision maker used to break ties. It will be applied only
        to the tied alternatives from the primary decision maker.

    force : bool, default True
        If True, when the untier decision maker also produces ties, uses
        the untied_rank_ property to force a complete ranking without ties.
        If False, allows the final ranking to have ties if the untier fails
        to break them completely.

    """

    _skcriteria_dm_type = "tie_breaker"
    _skcriteria_parameters = ["dmaker", "untier", "force"]

    def __init__(self, dmaker, untier, force=True):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        if not (hasattr(untier, "evaluate") and callable(untier.evaluate)):
            raise TypeError("'untier' must implement 'evaluate()' method")

        self._dmaker = dmaker
        self._untier = untier
        self._force = bool(force)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dec_repr = repr(self._dmaker)
        untier_repr = repr(self._untier)
        force = self._force
        return (
            f"<{name} dmaker={dec_repr}, untier={untier_repr}, force={force}>"
        )

    # PROPERTIES ==============================================================

    @property
    def dmaker(self):
        """Primary decision maker."""
        return self._dmaker

    @property
    def untier(self):
        """Secondary decision maker for breaking ties."""
        return self._untier

    @property
    def force(self):
        """Whether to force complete untying using untied_rank_."""
        return self._force

    # LOGIC ===================================================================

    def _generate_tied_groups(self, orank):
        values = set(orank.values)
        series = orank.to_series()
        for rank_value in sorted(values):
            alts = list(series[series == rank_value].index)
            yield rank_value, alts

    def _get_relative_rank_series(self, dm, alts, untier, last_assigned_rank):
        if len(alts) == 1:
            relative_rank_dict = {alts[0]: last_assigned_rank + 1}
        else:
            sub_dm = dm.loc[alts]
            sub_rank = untier.evaluate(sub_dm)
            relative_values = sub_rank.values + last_assigned_rank
            relative_rank_dict = dict(zip(alts, relative_values))

        relative_rank_series = pd.Series(relative_rank_dict)
        return relative_rank_series

    def _get_rank_method_name(self, orank, untier):
        omethod = orank.method
        untier_name = untier.get_method_name()
        method_name = f"{omethod}+TieBreaker({untier_name})"
        return method_name

    def _patch_extra(self, orank, untier, forced):
        extra = orank.extra_.to_dict()
        extra["tiebreaker"] = {
            "original_method": orank.method,
            "untier_method": untier.get_method_name(),
            "original_values": orank.values,
            "forced": forced,
        }
        return extra

    def evaluate(self, dm):
        """Evaluate the decision matrix using the untier approach.

        Parameters
        ----------
        dm : DecisionMatrix
            Decision matrix to evaluate.

        Returns
        -------
        result : RankResult
            Ranking result with ties broken using the untier decision maker.
        """
        # all as locals
        dmaker = self._dmaker
        untier = self._untier
        force = self._force

        # Get initial ranking from primary decision maker
        orank = dmaker.evaluate(dm)

        # If no ties, return original ranking
        if not orank.has_ties_:
            return orank

        last_assigned_rank = 0
        untied_rank_series = None
        for rank, alts in self._generate_tied_groups(orank):
            relative_rank_series = self._get_relative_rank_series(
                dm, alts, untier, last_assigned_rank
            )
            last_assigned_rank = np.max(relative_rank_series)
            untied_rank_series = pd.concat(
                [untied_rank_series, relative_rank_series],
            )

        untied_rank_series_sorted = untied_rank_series[orank.alternatives]

        method_name = self._get_rank_method_name(orank, untier)
        extra = self._patch_extra(orank, untier, forced=False)

        untied_rank = RankResult(
            method=method_name,
            alternatives=orank.alternatives,
            values=untied_rank_series_sorted.values,
            extra=extra,
        )

        if not untied_rank.has_ties_:
            return untied_rank
        elif force:
            forced_values = untied_rank.untied_rank_
            extra = self._patch_extra(orank, untier, forced=True)
            untied_rank = RankResult(
                method=method_name,
                alternatives=orank.alternatives,
                values=forced_values,
                extra=extra,
            )

        return untied_rank
