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

    def _find_tied_groups(self, ranking_result):
        """Identify and group alternatives that share the same ranking.

        This method analyzes a ranking result to find which alternatives are
        tied (have the same rank value). Uses pandas groupby operations for
        efficient grouping of alternatives by their rank values.

        Example
        -------
        For a ranking [1, 1, 2, 2, 3] with alternatives [A, B, C, D, E]:
        - Rank 1: alternatives {A, B} (tied)
        - Rank 2: alternatives {C, D} (tied)
        - Rank 3: alternative {E} (single alternative)

        Returns: {1: {A, B}, 2: {C, D}, 3: {E}}

        Parameters
        ----------
        ranking_result : RankResult
            The ranking result from the primary decision maker containing
            alternatives and their corresponding rank values.

        Returns
        -------
        tied_groups_dict : dict
            Dictionary mapping rank values (int) to sets of alternatives (str)
            that share that rank. Includes all ranks, even single alternatives.
        """
        # Convert ranking result to pandas Series (alternative -> rank_value)
        ranking_series = ranking_result.to_series()

        # Group alternatives by their rank values using pandas groupby
        # groupby(series) groups by the values in the series
        grouped_by_rank = ranking_series.groupby(ranking_series)

        # For each rank value, collect the alternatives (indices) that have
        # that rank apply(lambda g: set(g.index)) extracts the alternative
        # names for each group
        tied_groups_dict = grouped_by_rank.apply(
            lambda group: list(group.index)
        ).to_dict()

        return tied_groups_dict

    def _break_ties(self, dm, tied_groups, untier):
        """Break ties in rankings using a specified tie-breaking method.

        This method processes groups of tied alternatives and uses a
        tie-breaking method to determine a relative ranking without ties.

        Parameters
        ----------
        dm : pandas.DataFrame
            Decision matrix containing criteria and alternatives.
            Rows represent alternatives and columns represent criteria.
        tied_groups : dict
            Dictionary where keys are rank values and values are lists of
            alternatives that share that rank (ties).
        untier : object
            Object that implements a tie-breaking method.
            Must have an evaluate() method that takes a decision matrix and
            returns a ranking.

        Returns
        -------
        dict
            Dictionary with the relative ranking without ties, where keys are
            alternatives and values are their positions in the tied group.

        Notes
        -----
        The method processes each tied group separately:
        - If a group contains only one alternative, it keeps its original
          ranking
        - If a group contains multiple alternatives, it applies the
          tie-breaking method on a subset of the decision matrix that includes
          only those alternatives

        """
        # Dict to store final ranking without ties
        untied_relative_rankings = {}

        # Iterate over each tied group
        for current_rank_value, tied_alternatives_list in tied_groups.items():

            # If no tie exists (only one alternative), keep original ranking
            if len(tied_alternatives_list) == 1:
                untied_relative_rankings[tied_alternatives_list[0]] = (
                    current_rank_value
                )
                continue

            # Create submatrix with only tied alternatives
            tied_alternatives_submatrix = dm.loc[tied_alternatives_list]

            # Apply tie-breaking method to the submatrix
            untied_relative_ranking_result = untier.evaluate(
                tied_alternatives_submatrix
            )

            # Convert result to pandas series
            ranking_series = untied_relative_ranking_result.to_series()

            # Update final rankings dictionary
            untied_relative_rankings.update(ranking_series.to_dict())

        return untied_relative_rankings

    def _reconstruct_ranking(
        self, original_rank, tied_groups, untied_rankings
    ):
        """Reconstruct the final ranking by combining original ranks with
        tie-breaking results.

        Parameters
        ----------
        original_rank : RankResult
            Original ranking from primary decision maker.
        tied_groups : dict
            Dictionary of tied groups.
        untied_rankings : dict
            Dictionary with tie-breaking results.

        Returns
        -------
        final_values : array
            Final ranking values without ties.
        """
        alternatives = original_rank.alternatives
        original_values = original_rank.values
        final_values = np.zeros_like(original_values, dtype=int)

        coso = sorted(zip(original_values, alternatives))

        last_ranking = 0
        for idx, (orank, alt) in enumerate(coso):
            import ipdb

            ipdb.set_trace()

            # if orig_rank in tied_groups and alt in untied_rankings:
            #     # This alternative was tied and has been untied
            #     tied_alts = tied_groups[orig_rank]
            #     relative_rank = untied_rankings[alt]

            #     # Calculate final rank: original rank + relative position - 1
            #     final_rank = orig_rank + relative_rank - 1
            #     final_values[i] = final_rank
            # else:
            #     # Alternative was not tied, keep original rank
            #     final_values[i] = orig_rank

        return final_values

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

        # Get initial ranking from primary decision maker
        original_rank = dmaker.evaluate(dm)

        # If no ties, return original ranking
        if not original_rank.has_ties_:
            return original_rank

        # Find tied groups
        tied_groups = self._find_tied_groups(original_rank)

        # Break ties using untier decision maker
        untied_rankings = self._break_ties(dm, tied_groups, untier)

        # Reconstruct final ranking
        final_values = self._reconstruct_ranking(
            original_rank, tied_groups, untied_rankings
        )

        # Create result
        method_name = f"{original_rank.method}+TieBreaker({self._untier.get_method_name()})"

        # Prepare extra information
        extra = dict(original_rank.extra_.items())
        extra["untier_info"] = {
            "original_method": original_rank.method,
            "untier_method": self._untier.get_method_name(),
            "tied_groups": tied_groups,
            "force_used": self._force,
        }

        result = RankResult(
            method=method_name,
            alternatives=original_rank.alternatives,
            values=final_values,
            extra=extra,
        )

        return result
