#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of the rank reversal test by cloning alternatives.

This module provides the `RankClonesChecker` class, which evaluates the
stability of an MCDA method by introducing copies of non-optimal alternatives
and checking if the top-ranked alternative changes.

According to this criterion, the indication of the best alternative should
remain unchanged when identical or near-identical copies of non-optimal
alternatives are introduced.

"""

# =============================================================================
# imports
# =============================================================================

import numpy as np

import skcriteria as skc

from ..agg import RankResult
from ..cmp.ranks_cmp import RanksComparator
from ..core import SKCMethodABC
from ..utils import Bunch, unique_names
from ..utils.rank import is_rank


# =============================================================================
# CHECKER
# =============================================================================


class RankClonesChecker(SKCMethodABC):
    """Check the stability of a method against the cloning of alternatives.

    This checker evaluates if the best alternative remains the same when
    a copy of some of the non-optimal alternatives is introduced.

    The checker iterates through all non-optimal alternatives, creates a
    clone of each one, and evaluates the new decision matrix with the same
    decision maker. The result is a collection of rankings, one for each
    cloned alternative, which can be compared with the original ranking.

    Parameters
    ----------
    dmaker: Decision maker - must implement the ``evaluate()`` method
        The MCDA method, or pipeline to evaluate.

    """

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = ["dmaker"]

    def __init__(self, dmaker):
        # Validate that the dmaker has an evaluate method
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

    @property
    def dmaker(self):
        """The MCDA method, or pipeline to evaluate."""
        return self._dmaker

    def _get_suboptimal_alternatives(self, rank):
        """Extract the suboptimal alternatives from a ranking."""
        series = rank.to_series()
        return series.index[series > 1]

    def _dm_with_clone(self, dm, original_alternative):
        """Create a new decision matrix with a cloned alternative."""
        # create the name of the clone
        cloned_alternative_name = f"D.{original_alternative}"

        # clone the alternative
        cloned_matrix = dm.matrix.copy()
        cloned_matrix.loc[cloned_alternative_name] = dm.matrix.loc[
            original_alternative
        ].copy()

        # create the new decision matrix
        cloned_dm = dm.replace(
            matrix=cloned_matrix.to_numpy(),
            alternatives=cloned_matrix.index.to_numpy(),
        )

        return cloned_alternative_name, cloned_dm

    def _patch_rank(
        self,
        rank,
        cloned_alternative,
        cloned_alternative_name,
    ):
        """Add information about the cloning process to the ranking."""
        # extract the original data
        method = str(rank.method)
        alternatives = rank.alternatives.copy()
        values = rank.values.copy()
        extra = dict(rank.extra_.items())
        cloned_alternative_value = None
        rank_shifted = False

        # If this is a cloned rank, we proceed to remove the clone from the
        # ranking and fix the ranks if necessary.
        if cloned_alternative is not None:
            # Change the method name for clarity in the final comparator
            method = f"{method}+CloneCheck+{cloned_alternative}"

            # Find the clone in the ranking and separate it
            preserve_alternatives = np.argwhere(
                alternatives != cloned_alternative_name
            ).flatten()

            cloned_alternative_idx = np.argwhere(
                alternatives == cloned_alternative_name
            ).flatten()[0]

            # Store the rank value of the clone before removing it
            cloned_alternative_value = values[cloned_alternative_idx]

            # Remove the clone from the alternatives and values arrays
            alternatives = alternatives[preserve_alternatives]
            values = values[preserve_alternatives]

            # IMPORTANT: If removing the clone creates a gap in the ranks,
            # we need to shift all subsequent ranks down by one to keep it dense.
            # e.g., a rank [1, 3, 4] becomes [1, 2, 3]
            if not is_rank(values):
                values[values > cloned_alternative_value] -= 1
                rank_shifted = True

        # We create a new bunch with all the information of the cloning
        # process to be able to analyze it later.
        extra["rank_clone_check"] = Bunch(
            "rank_clone_check",
            {
                "cloned_alternative": cloned_alternative,
                "cloned_alternative_name": cloned_alternative_name,
                "cloned_alternative_value": cloned_alternative_value,
                "original_rank": rank,
                "rank_shifted": rank_shifted,
            },
        )

        # return the new rank result
        patched_rank = RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
        return patched_rank

    def evaluate(self, dm):
        """Execute the checker.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix to be evaluated.

        Returns
        -------
        RanksComparator
            An object containing multiple rankings of the alternatives, with
            information on any changes made to the original decision matrix in
            the `extra_` attribute.

        """
        # 1. Calculate the original ranking to serve as a reference.
        dmaker = self.dmaker
        original_rank = dmaker.evaluate(dm)
        patched_original_rank = self._patch_rank(
            rank=original_rank,
            cloned_alternative=None,
            cloned_alternative_name=None,
        )

        # 2. Identify which alternatives are not the best.
        suboptimal_alternatives = self._get_suboptimal_alternatives(
            original_rank
        )

        # 3. For each suboptimal alternative, create a clone and evaluate.
        # Here we create a containers for the rank comparator starting with
        # the reference rank
        names, results = ["Reference"], [patched_original_rank]
        for alternative_to_clone in suboptimal_alternatives:

            # create the new decision matrix with the clone
            cloned_alternative_name, cloned_dm = self._dm_with_clone(
                dm, alternative_to_clone
            )

            # Evaluate the new dm
            cloned_rank = dmaker.evaluate(cloned_dm)

            # Add info about the clone to the rank
            patched_cloned_rank = self._patch_rank(
                rank=cloned_rank,
                cloned_alternative=alternative_to_clone,
                cloned_alternative_name=cloned_alternative_name,
            )

            # store the information
            names.append(cloned_alternative_name)
            results.append(patched_cloned_rank)

        # 4. Collect and return all rankings in a comparator object.
        named_ranks = unique_names(names=names, elements=results)
        return RanksComparator(named_ranks, extra={})