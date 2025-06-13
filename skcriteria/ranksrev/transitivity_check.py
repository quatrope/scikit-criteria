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
from ..utils import Bunch, rank, generate_acyclic_graphs, unique_names


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

# TODO: discuss if this should move into a sepparate module


def _untie_first(r1, r2):
    return [(r1, r2)]


def _untie_second(r1, r2):
    return [(r2, r1)]


def _untie_equivalent_ranks(r1, r2):
    # The untie criteria is non-trivial and greatly affects the resulting graph.
    # This default criteria inserts them with arbitrary order, but other
    # heuristics could work.
    # 1.   Inserting both ranks with a different order.
    # 2-3. Insert only one rank.
    # 4.   Insert both ranks in different orders (causing a cycle).
    # 5.   Insert none (causing a disjount graph).
    return [(r1, r2), (r2, r1)]


def _untie_by_dominance(r1, r2):
    dominance_result = rank.dominance(r1, r2)
    winner_edge = (
        (r1, r2) if dominance_result.aDb < dominance_result.bDa else (r2, r1)
    )
    return [winner_edge]


def _transitivity_break_bound(n):
    """
    Calculate the maximum number of transitivity violations possible in a n-tournament.

    This function computes the theoretical upper bound for the number of 3-cycles
    (intransitive triples) that can occur in a tournament with n alternatives.
    A 3-cycle occurs when alternative A beats B, B beats C, but C beats A,
    violating transitivity.

    Parameters
    ----------
    n : int
        Number of alternatives/participants in the tournament.
        Must be a positive integer >= 3 for meaningful results.

    Returns
    -------
    int
        Maximum possible number of transitivity violations (3-cycles) in a
        tournament of size n. Returns 0 for n < 3.

    Notes
    -----
    This bound represents the worst-case scenario for transitivity violations.

    References
    ----------
    :cite:p:`roy1990outranking`
    """
    if n % 2 == 0:
        return n * (n**2 - 4) // 24
    else:
        return n * (n**2 - 1) // 24


<<<<<<< HEAD
def _both_bound(n):
    return n * (n - 1) * (n - 2) / 3


=======
>>>>>>> 9468ece029d5ef26a09b0a8bdace5f4982792ad1
def in_degree_sort(dag):
    """
    Sorts the nodes of a directed acyclic graph (DAG) into hierarchical groups
    based on in-degree using the graph's transitive reduction.

    The result represents a topological layering of the graph.

    Parameters:
    -----------
    dag : networkx.DiGraph
        A directed acyclic graph (DAG).

    Returns:
    --------
    list[list[hashable]]
        A list of lists of nodes grouped by their in-degree level.
    """
    graph_reduction = nx.transitive_reduction(dag)
    groups_sort = []

    while graph_reduction.nodes:
        group = [
            node
            for node in list(graph_reduction.nodes)
            if graph_reduction.in_degree(node) == 0
        ]
        groups_sort.append(group)
        graph_reduction.remove_nodes_from(group)

    return groups_sort


def assign_rankings(groups):
    """
    Assign ascending integer rankings to grouped items.

    All items in the same group share the same rank, starting from 1 for the first group,
    and increasing by 1 for each subsequent group.

    Parameters
    ----------
    groups : list[list[hashable]]
        A list of groups (each group is a list of items).

    Returns
    -------
    dict
        A dictionary mapping each item to its assigned rank.
    """
    rankings = {}
    for rank, group in enumerate(groups, start=1):
        for item in group:
            rankings[item] = rank

    return rankings


# =============================================================================
# INTERNAL VARIABLES
# =============================================================================

_PAIR_RANK_UNTIERS = {
    "both": {
        "edge_creation": _untie_equivalent_ranks,
        "bound_function": _both_bound,
    },
    "dominance": {
        "edge_creation": _untie_by_dominance,
        "bound_function": _transitivity_break_bound,
    },
    "first": {
        "edge_creation": _untie_first,
        "bound_function": _transitivity_break_bound,
    },
    "second": {
        "edge_creation": _untie_second,
        "bound_function": _transitivity_break_bound,
    },
}

# =============================================================================
# CLASS
# =============================================================================


class TransitivityChecker(SKCMethodABC):
    """
    Parameters
    ----------
    dmaker: Decision maker - must implement the ``evaluate()`` method
        The MCDA method, or pipeline to evaluate.

    parallel_backend: str or None (default: None)
        Evaluate alternatives using multithreading, multiprocessing, or
        sequential computation

    random_state: int, numpy.random.default_rng or None (default: None)
        Controls the random state to generate variations in the sub-optimal
        alternatives.

    pair_rank_untier: callable or None (default: None)
        Function that determines the order of 2 equivalent rankings.
        Must return a sequence of pairs.
    """

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = [
        "dmaker",
        "parallel_backend",
        "random_state",
    ]

    def __init__(
        self,
        dmaker,
        *,
        pair_rank_untier="both",
        random_state=None,
        cycle_removal_strategy="random",
        max_acyclic_graphs=10000,
        parallel_backend=None,
        n_jobs=None,
    ):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        # UNTIE EQUIVALENT RANKS
        self._pair_rank_untier = pair_rank_untier

        # PARALLEL BACKEND
        self._parallel_backend = parallel_backend
        self._n_jobs = n_jobs

        # RANDOM
        self._random_state = np.random.default_rng(random_state)

        # STRATEGY FOR REMOVAL OF CYCLES
        self._cycle_removal_strategy = (
            cycle_removal_strategy  # TODO VERR CONDICION
        )

        # MAXIMIMUM PERMITED ACYCLIC GRAPHS TO BE GENERATED
        self._max_acyclic_graphs = max_acyclic_graphs  # TODO VER CONDICIONN

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dm = repr(self.dmaker)
        # repeats = self.repeat
        # ama = self._allow_missing_alternatives
        # lds = self.last_diff_strategy
        return (
            f"<{name} {dm}>"
            #   repeats={repeats}, "
            # f"allow_missing_alternatives={ama} last_diff_strategy={lds!r}>"
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

    @property
    def cycle_removal_strategy(self):
        return self._cycle_removal_strategy

    @property
    def max_acyclic_graphs(self):
        return self._max_acyclic_graphs

    @property
    def n_jobs(self):
        return self._n_jobs

    # LOGIC ===================================================================

    def _evaluate_pairwise_submatrix(
        self, decision_matrix, alternative_pair, pipeline
    ):
        """
        Apply the MCDM pipeline to a sub-problem of two alternatives
        """
        sub_dm = decision_matrix.loc[alternative_pair]
        return pipeline.evaluate(sub_dm)

    def _get_graph_edges(self, results):
        """
        Generate directed graph edges from pairwise comparison results.

        Parameters
        ----------
        results : iterable
            Collection of comparison result objects. Each result must contain:
            - alternatives : list or tuple
                Names/identifiers of the two compared alternatives
            - rank_ : list or array-like
                Ranking values for each alternative (lower values indicate better ranking)

        Returns
        -------
        list
            List of tuples (winner, loser) representing directed edges in the preference graph.
            Each tuple indicates that the first alternative is preferred over the second.
            For tied rankings, applies tie-breaking logic via _pair_rank_untier() method.

        Notes
        -----
        - Uses lower-is-better ranking system (rank 1 > rank 2 > rank 3)
        - Automatically handles tied rankings through internal tie-breaking mechanism
        - Output format is suitable for constructing tournament or preference graphs
        """
        edges = []

        # Get the rank untier strategy
        pair_rank_untier = self._pair_rank_untier
        tie_breaker = _PAIR_RANK_UNTIERS[pair_rank_untier]["edge_creation"]

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
                untied_edges = tie_breaker(alt_names[0], alt_names[1])
                edges.extend(untied_edges)

        return edges

    def _create_rank_with_info(self, orank, extra, dag, edges):

        # topological_sorts = list(nx.all_topological_sorts(dag))
        # sort_count = len(topological_sorts)

        # if sort_count > 1:
        alternative_rank_value = assign_rankings(in_degree_sort(dag))

        # else:
        #     sorted_alternatives = list(nx.lexicographical_topological_sort(dag))

        #     alternative_rank_value = dict(
        #         zip(
        #             sorted_alternatives,
        #             np.arange(1,len(sorted_alternatives)+1),
        #         )
        #     )

        print(f"Tipo de extra: {type(extra)}")

        untied_rank = RankResult(
            method=orank.method,
            alternatives=orank.alternatives,
            values=np.array(
                [alternative_rank_value[alt] for alt in orank.alternatives]
            ),
            extra=extra,
        )

        return untied_rank

    def _get_ranks(self, graph, orank, extra):

        untied_ranks = []

        acyclic_graphs = generate_acyclic_graphs(
            graph,
            strategy=self._cycle_removal_strategy,
            max_graphs=self._max_acyclic_graphs,
            seed=self._random_state,
        )

        for (
            dag,
            edges,
        ) in acyclic_graphs:
            untied_rank = self._create_rank_with_info(orank, extra, dag, edges)
            untied_ranks.append(untied_rank)

        return list(untied_ranks)

    def _dominance_graph(self, dm, orank):
        # Generate all pairwise combinations of alternatives
        # For n alternatives, creates C(n,2) = n*(n-1)/2 unique sub-problems
        pairwise_combinations = map(
            list, it.combinations(orank.alternatives, 2)
        )

        dmaker = self._dmaker

        # Parallel processing of all pairwise sub-matrices
        # Each resulting sub-matrix has 2 alternatives × k original criteria
        with joblib.Parallel(
            prefer=self._parallel_backend, n_jobs=self._n_jobs
        ) as P:
            delayed_evaluation = joblib.delayed(
                self._evaluate_pairwise_submatrix
            )
            results = P(
                delayed_evaluation(dm, pair, dmaker)
                for pair in pairwise_combinations
            )

        edges = self._get_graph_edges(results)

        # Create directed graph
        return nx.DiGraph(edges)

    def _calculate_transitivity_break(self, graph):

        # TODO: Justificar el 3 en length_bound
        trans_break = list(nx.simple_cycles(graph, length_bound=3))

        trans_break_rate = len(trans_break) / _transitivity_break_bound(
            len(graph.nodes)
        )

        return trans_break, trans_break_rate

    def _generate_graph_data(self, dm, orank):
        # Create pairwise dominance graph
        graph = self._dominance_graph(dm, orank)

        # Calculate transitivity break, and it's rate
        trans_break, trans_break_rate = self._calculate_transitivity_break(
            graph
        )
        return graph, trans_break, trans_break_rate

    def _test_criterion_3(self, graph):
        """Perform test criterion 3"""
        cycles = nx.simple_cycles(graph)
        cycles_count = len(np.asarray([c for c in cycles if len(c) >= 3]))
        return cycles_count == 0

    def evaluate(self, *, dm):
        """Executes the transitivity test.

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
            contains an object in the key `rrt1` that provides
            information on any changes made to the original decision matrix,
            including the noise applied to worsen any suboptimal
            alternative.

        """

        dmaker = self._dmaker

        # we need a first reference ranking
        orank = dmaker.evaluate(dm)

        extra = orank.extra_

        graph, trans_break, trans_break_rate = self._generate_graph_data(
            dm, orank
        )

        # TODO: What is test criterion 3?
        test_criterion_3 = self._test_criterion_3(graph)

        returned_ranks = []
        returned_ranks = self._get_ranks(graph, orank, extra)

        names = ["Original"] + [
            f"Untied{i+1}" for i in range(len(returned_ranks))
        ]

        named_ranks = unique_names(
            names=names, elements=[orank] + returned_ranks
        )

        return RanksComparator(
            named_ranks,
            extra={
                "pairwise_dominance_graph": graph,
                "transitivity_breaks": trans_break,
                "transitivity_break_rate": trans_break_rate,
                "test_criterion_3": test_criterion_3,
            },
        )
