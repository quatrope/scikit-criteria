#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tools for evaluating the stability of MCDA method's best alternative.

According to this criterion, dividing the alternatives by pair should
keep transitivity when grouped again.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

import joblib

import networkx as nx

import numpy as np

import pandas as pd

from ..agg import RankResult
from ..cmp import RanksComparator
from ..core import SKCMethodABC
from ..utils import Bunch, break_cycles_greedy, unique_names


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

# TODO: discuss if this should move into a sepparate module

def _untie_equivalent_ranks(r1, r2):
    # The untie criteria is non-trivial and greatly affects the resulting graph.
    # This default criteria inserts them with arbitrary order, but other
    # heuristics could work.
    # 1.   Inserting both ranks with a different order.
    # 2-3. Insert only one rank.
    # 4.   Insert both ranks in different orders (causing a cycle).
    # 5.   Insert none (causing a disjount graph).
    return ((r1, r2), (r2, r1))

# =============================================================================
# CLASS
# =============================================================================


class TransitivityChecker(SKCMethodABC):
    r""""""

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = [
        "dmaker",
        "parallel_backend",
        "random_state",
    ]

    def __init__(self, dmaker, *, parallel_backend=None, random_state=None, pair_rank_untier=None):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        # UNTIE EQUIVALENT RANKS
        if pair_rank_untier and not callable(pair_rank_untier):
            raise TypeError("'pair_rank_untier' must be callable")
        self._pair_rank_untier = pair_rank_untier or _untie_equivalent_ranks

        # PARALLEL BACKEND
        self._parallel_backend = parallel_backend

        # RANDOM
        self._random_state = np.random.default_rng(random_state)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dm = repr(self.dmaker)
        repeats = self.repeat
        ama = self._allow_missing_alternatives
        lds = self.last_diff_strategy
        return (
            f"<{name} {dm} repeats={repeats}, "
            f"allow_missing_alternatives={ama} last_diff_strategy={lds!r}>"
        )

    # PROPERTIES ==============================================================

    @property
    def dmaker(self):
        """The MCDA method, or pipeline to evaluate."""
        return self._dmaker

    @property
    def parallel_backend(self):
        return self._parallel_backend

    @property
    def random_state(self):
        """Controls the random state to generate variations in the \
        sub-optimal alternatives."""
        return self._random_state

    # LOGIC ===================================================================

    def _evaluate_pairwise_submatrix(
        self, decision_matrix, alternative_pair, pipeline
    ):
        """
        Apply the MCDM pipeline to a sub-problem of two alternatives
        """
        sub_dm = decision_matrix.loc[alternative_pair]
        return pipeline.evaluate(sub_dm)

    def evaluate(self, dm):
        """Executes a the transitivity test.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix to be evaluated.

        Returns
        -------
        RanksComparator
            An object containing multiple rankings of the alternatives, with
            information on any changes made to the original decision matrix in
            the `extra_` attribute. Specifically, the `extra_` attribute
            contains a an object in the key `rrt1` that provides
            information on any changes made to the original decision matrix,
            including the the noise applied to worsen any sub-optimal
            alternative.

        """
        dmaker = self._dmaker

        # we need a first reference ranking
        original_rank = dmaker.evaluate(dm)

        # Generate all pairwise combinations of alternatives
        # For n alternatives, creates C(n,2) = n*(n-1)/2 unique sub-problems
        pairwise_combinations = map(list, it.combinations(dm.alternatives, 2))

        # Parallel processing of all pairwise sub-matrices
        # Each resulting sub-matrix has 2 alternatives × k original criteria
        with joblib.Parallel(prefer=self._parallel_backend) as P:
            delayed_evaluation = joblib.delayed(self._evaluate_pairwise_submatrix)
            results = P(
                delayed_evaluation(dm, pair, dmaker) for pair in pairwise_combinations
            )

        edges = []

        # TODO: move this to a different function
        for rr in results:
            # Access the names of the compared alternatives
            alt_names = rr.alternatives

            # Access the ranking assigned by the model
            ranks = rr.rank_

            # Identify which one is ranked better (lower number is better)
            if ranks[0] < ranks[1]:
                edges.append((alt_names[0], alt_names[1]))
            elif ranks[1] < ranks[0]:
                edges.append((alt_names[1], alt_names[0]))
            else:
                untied_ranks = self._pair_rank_untier(alt_names[0], alt_names[1])
                if untied_ranks:
                    edges.extend(untied_ranks)

        # TODO: Untie between ranking (topological sort).
        # Heuristics to break cycles.
        # KEEP THE ORIGINAL GRAPH!!!!!!!!!!!!!!!!!!!!

        # Create directed graph
        G = nx.DiGraph()
        G.add_edges_from(edges)

        cycles = nx.recursive_simple_cycles(G)
        acyclic_graph = break_cycles_greedy(G)

        sorted_rank = list(nx.topological_sort(acyclic_graph))

        extra = dict(original_rank.extra_.items())
        extra["rrt2"] = Bunch(
            "rrt2",
            {
                "original_graph": G,
                "cycles": cycles,
                "acyclic_graph": acyclic_graph,
                "sorted_rank": sorted_rank
            }
        )

        untied_rank = RankResult(
            method=original_rank.method,
            alternatives=sorted_rank,
            values=np.arange(1, len(sorted_rank)+1),
            extra=extra,
        )

        named_ranks = unique_names(names=["Original", "Untied"], elements=[original_rank, untied_rank])
        return RanksComparator(named_ranks)
