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


import numpy as np

import pandas as pd


from .agg import RankResult
from .core import SKCMethodABC
from .utils.bunch import Bunch

# =============================================================================
# CLASS
# =============================================================================


class TieBreaker(SKCMethodABC):
    """Decision maker that breaks ties in rankings using a secondary \
    decision maker.

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

    Examples
    --------
    >>> from skcriteria import mkdm
    >>> from skcriteria.madm import simple
    >>>
    >>> # Create a decision matrix
    >>> dm = mkdm([[1, 2], [3, 4]], [1, 1], ["max", "max"])
    >>>
    >>> # Primary decision maker
    >>> primary = simple.WeightedSum()
    >>>
    >>> # Secondary decision maker for tie breaking
    >>> secondary = simple.WeightedProduct()
    >>>
    >>> # Create tie breaker
    >>> tie_breaker = TieBreaker(primary, secondary)
    >>>
    >>> # Evaluate
    >>> result = tie_breaker.evaluate(dm)
    """

    _skcriteria_dm_type = "tie_breaker"
    _skcriteria_parameters = ["dmaker", "untier", "force"]

    def __init__(self, dmaker, untier, *, force=True):
        # Validate that both decision makers implement the evaluate method
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        if not (hasattr(untier, "evaluate") and callable(untier.evaluate)):
            raise TypeError("'untier' must implement 'evaluate()' method")

        self._dmaker = dmaker
        self._untier = untier
        self._force = bool(force)

    def __repr__(self):
        """Return string representation of the TieBreaker instance.

        Returns
        -------
        str
            String representation showing the primary decision maker,
            untier decision maker, and force parameter.
        """
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
        """Primary decision maker.

        Returns
        -------
        decision maker
            The primary decision maker instance used for initial ranking.
        """
        return self._dmaker

    @property
    def untier(self):
        """Secondary decision maker for breaking ties.

        Returns
        -------
        decision maker
            The secondary decision maker instance used to break ties
            from the primary decision maker.
        """
        return self._untier

    @property
    def force(self):
        """Whether to force complete untying using untied_rank_.

        Returns
        -------
        bool
            True if forced untying is enabled, False otherwise.
        """
        return self._force

    # LOGIC ===================================================================

    def _generate_tied_groups(self, orank):
        """Generate groups of alternatives that share the same rank value.

        This method identifies all alternatives that have tied rankings
        and yields them grouped by their rank value.

        Parameters
        ----------
        orank : RankResult
            Original ranking result from the primary decision maker.

        Yields
        ------
        tuple of (int, list)
            A tuple containing the rank value and a list of alternatives
            that share that rank.
        """
        # Get all unique rank values
        values = set(orank.values)
        series = orank.to_series()

        # For each rank value, find all alternatives with that rank
        for rank_value in sorted(values):
            alts = list(series[series == rank_value].index)
            yield rank_value, alts

    def _get_relative_rank_series(self, dm, alts, untier, last_assigned_rank):
        """Calculate relative ranks for a group of tied alternatives.

        For a given group of tied alternatives, this method either assigns
        the next sequential rank (if only one alternative) or uses the
        untier decision maker to break ties within the group.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix containing all alternatives and criteria.
        alts : list
            List of alternative indices that are tied.
        untier : decision maker
            The untier decision maker to use for breaking ties.
        last_assigned_rank : int
            The last rank value that was assigned to maintain sequential
            ranking.

        Returns
        -------
        pandas.Series
            Series containing the relative ranks for the tied alternatives,
            indexed by alternative names.
        """
        if len(alts) == 1:
            # Single alternative, assign next sequential rank
            relative_rank_dict = {alts[0]: last_assigned_rank + 1}
        else:
            # Multiple tied alternatives, use untier to break ties
            sub_dm = dm.loc[alts]  # Extract submatrix for tied alternatives
            sub_rank = untier.evaluate(sub_dm)  # Apply untier decision maker

            # Adjust ranks to maintain sequential ordering
            relative_values = sub_rank.values + last_assigned_rank
            relative_rank_dict = dict(zip(alts, relative_values))

        relative_rank_series = pd.Series(relative_rank_dict)
        return relative_rank_series

    def _get_rank_method_name(self, orank, untier):
        """Construct the method name for the tie-broken ranking.

        Creates a descriptive method name that shows both the original
        ranking method and the untier method used for tie breaking.

        Parameters
        ----------
        orank : RankResult
            Original ranking result containing the method name.
        untier : decision maker
            The untier decision maker instance.

        Returns
        -------
        str
            Formatted method name showing the combination of methods used.
        """
        omethod = orank.method
        untier_name = untier.get_method_name()
        method_name = f"{omethod}+TieBreaker({untier_name})"
        return method_name

    def _patch_extra(self, orank, untier, forced):
        """Create enhanced metadata dictionary for the tie-broken result.

        Combines the original ranking metadata with additional information
        about the tie-breaking process.

        Parameters
        ----------
        orank : RankResult
            Original ranking result containing existing metadata.
        untier : decision maker
            The untier decision maker instance.
        forced : bool
            Whether forced untying was applied to eliminate remaining ties.

        Returns
        -------
        dict
            Enhanced metadata dictionary containing original information
            plus tie-breaking details.
        """
        # Start with original metadata
        extra = orank.extra_.to_dict()

        # Add tie-breaking specific information
        extra["tiebreaker"] = Bunch(
            "tiebreaker",
            {
                "original_method": orank.method,
                "untier_method": untier.get_method_name(),
                "original_values": orank.values,
                "forced": forced,
            },
        )
        return extra

    def evaluate(self, dm):
        """Evaluate the decision matrix using the tie-breaking approach.

        This method first applies the primary decision maker to get an initial
        ranking. If ties exist, it systematically applies the untier decision
        maker to each group of tied alternatives to break the ties.

        Parameters
        ----------
        dm : DecisionMatrix
            Decision matrix to evaluate containing alternatives and criteria.

        Returns
        -------
        result : RankResult
            Ranking result with ties broken using the untier decision maker.
            If force=True and ties still remain, uses untied_rank_ to ensure
            a complete ranking without any ties.
        """
        # Store instance variables as locals for efficiency
        dmaker = self._dmaker
        untier = self._untier
        force = self._force

        # Get initial ranking from primary decision maker
        orank = dmaker.evaluate(dm)

        # Early return if no ties exist in the original ranking
        if not orank.has_ties_:
            return orank

        # Initialize variables for tie-breaking process
        last_assigned_rank = 0
        untied_rank_series = None

        # Process each group of tied alternatives
        for rank, alts in self._generate_tied_groups(orank):
            # Get relative ranks for this tied group
            relative_rank_series = self._get_relative_rank_series(
                dm, alts, untier, last_assigned_rank
            )

            # Update the last assigned rank for sequential ordering
            last_assigned_rank = np.max(relative_rank_series)

            # Concatenate with previous results
            untied_rank_series = pd.concat(
                [untied_rank_series, relative_rank_series],
            )

        # Reorder to match original alternative ordering
        untied_rank_series_sorted = untied_rank_series[orank.alternatives]

        # Build method name and metadata
        method_name = self._get_rank_method_name(orank, untier)
        extra = self._patch_extra(orank, untier, forced=False)

        # Create the tie-broken ranking result
        untied_rank = RankResult(
            method=method_name,
            alternatives=orank.alternatives,
            values=untied_rank_series_sorted.values,
            extra=extra,
        )

        # Check if ties were successfully broken
        if not untied_rank.has_ties_:
            return untied_rank
        elif force:
            # Force complete untying if ties still remain and force=True
            forced_values = untied_rank.untied_rank_
            extra = self._patch_extra(orank, untier, forced=True)
            untied_rank = RankResult(
                method=method_name,
                alternatives=orank.alternatives,
                values=forced_values,
                extra=extra,
            )

        return untied_rank
