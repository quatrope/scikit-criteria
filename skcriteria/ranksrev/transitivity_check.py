#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Tools for evaluating the stability of MCDA method's best alternative.

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

from ..agg import RankResult
from ..cmp import RanksComparator
from ..core import SKCMethodABC
from ..utils import Bunch, generate_acyclic_graphs, rank, unique_names
from ..utils.cycle_removal import CYCLE_REMOVAL_STRATEGIES


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _untie_by_dominance(alt1, alt2, dm):
    """
    Resolve ties between two alternatives using dominance analysis.

    This function determines which of two alternatives dominates the other
    by comparing their performance across all criteria using pairwise
    dominance relationships. It helps resolve ties when alternatives have
    equal scores but one clearly dominates the other.

    Parameters
    ----------
    alt1 : hashable
        Identifier for the first alternative to compare.
    alt2 : hashable
        Identifier for the second alternative to compare.
    dm : DecisionMatrix
        The decision matrix containing criteria values for both alternatives.
        Must have both alt1 and alt2 as valid alternative indices.

    Returns
    -------
    list
        A list containing the dominance result:
        - If one alternative dominates: [(winner, loser)] where winner
          dominates loser
        - If no clear dominance exists: [] (empty list)

    Notes
    -----
    The function uses the dominance analysis framework to compare alternatives:

    1. Extract criteria values for both alternatives from the decision matrix
    2. Apply pairwise dominance analysis using `rank.dominance()`
    3. Compare dominance scores (aDb vs bDa) to determine winner
    4. Return the winning pair or empty list if tied

    The dominance relationship is asymmetric: if A dominates B more than
    B dominates A (aDb > bDa), then A is considered the winner.

    This function is typically used in tie-breaking scenarios where
    traditional scoring methods produce equal results but dominance
    analysis can reveal a preference.

    Examples
    --------
    >>> # Assuming a decision matrix with alternatives 'A' and 'B'
    >>> result = _untie_by_dominance('A', 'B', decision_matrix)
    >>> if result:
    ...     winner, loser = result[0]
    ...     print(f"{winner} dominates {loser}")
    ... else:
    ...     print("No clear dominance relationship")
    """
    crit1, crit2 = dm.alternatives[alt1], dm.alternatives[alt2]
    dominance_result = rank.dominance(crit1, crit2)
    aDb, bDa = dominance_result.aDb, dominance_result.bDa

    if aDb != bDa:
        winner = (alt1, alt2) if aDb > bDa else (alt2, alt1)
    else:
        return []

    return [winner]


def _transitivity_break_bound(n):
    """
    Calculate the maximum number of transitivity violations possible in a \
        n-tournament.

    This function computes the theoretical upper bound for the number of
    3-cycles (intransitive triples) that can occur in a tournament with n
    alternatives. A 3-cycle occurs when alternative A beats B, B beats C, but
    C beats A, violating transitivity.

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
    :cite:p:`moon2015topics`
    """
    if n % 2 == 0:
        return n * (n**2 - 4) // 24
    else:
        return n * (n**2 - 1) // 24


def _in_degree_sort(dag):
    """
    Sort nodes of a DAG into hierarchical groups based on in-degree levels.

    This function performs a topological layering of a directed acyclic graph
    by grouping nodes according to their in-degree in the graph's transitive
    reduction. Nodes with the same in-degree level represent alternatives
    at the same hierarchical level in the preference structure.

    Parameters
    ----------
    dag : networkx.DiGraph
        A directed acyclic graph (DAG) representing preference relationships
        between alternatives.

    Returns
    -------
    list of list
        A list where each element is a list of nodes at the same hierarchical
        level. The first sublist contains nodes with in-degree 0 (top level),
        the second contains nodes with in-degree 1 after removing the first
        level, and so on.

    Notes
    -----
    The algorithm works iteratively:

    1. Compute the transitive reduction of the input DAG to remove redundant
        edges
    2. Find all nodes with in-degree 0 (no incoming edges) - these form the
        top level
    3. Remove these nodes from the graph
    4. Repeat until all nodes are processed

    Examples
    --------
    >>> import networkx as nx
    >>>
    >>> # Create a simple DAG
    >>> dag = nx.DiGraph([('A', 'C'), ('B', 'C'), ('C', 'D')])
    >>> groups = _in_degree_sort(dag)
    >>> print(groups)
    [['A', 'B'], ['C'], ['D']]
    >>>
    >>> # A and B are at the top level (no predecessors)
    >>> # C is at level 2 (depends on A and B)
    >>> # D is at level 3 (depends on C)
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


def _assign_rankings(groups):
    """
    Assign ascending integer rankings to grouped items.

    This function converts hierarchical groups of items into a ranking
    dictionary where all items in the same group receive the same rank.
    Rankings start from 1 for the first group and increment by 1 for
    each subsequent group.

    Parameters
    ----------
    groups : list of list
        A list of groups where each group is a list of items that should
        receive the same ranking. Groups are processed in order, with
        earlier groups receiving better (lower) ranks.

    Returns
    -------
    dict
        A dictionary mapping each item to its assigned integer rank.
        Items in the first group get rank 1, items in the second group
        get rank 2, and so on.

    Notes
    -----
    This function is typically used after hierarchical sorting (like
    `_in_degree_sort`) to convert topological levels into numerical
    rankings. The ranking scheme assigns:

    - Rank 1: Best performing items (first group)
    - Rank 2: Second best items (second group)
    - And so on...

    All items within the same group are considered tied and receive
    identical ranks. This preserves the hierarchical structure while
    providing a simple numerical ranking.

    Examples
    --------
    >>> groups = [['A', 'B'], ['C'], ['D', 'E']]
    >>> rankings = _assign_rankings(groups)
    >>> print(rankings)
    {'A': 1, 'B': 1, 'C': 2, 'D': 3, 'E': 3}
    >>>
    >>> # A and B are tied for first place (rank 1)
    >>> # C is second (rank 2)
    >>> # D and E are tied for third place (rank 3)
    """
    rankings = {}
    for rank_number, group in enumerate(groups, start=1):
        for item in group:
            rankings[item] = rank_number

    return rankings


def _format_transitivity_cycles(cycles):
    """
    Format transitivity violation cycles for human-readable display.

    This function converts a list of cycles (representing transitivity
    violations) into a standardized string format that clearly shows
    the circular preference relationships. Each cycle is formatted to
    show the complete circular dependency.

    Parameters
    ----------
    cycles : list of list
        A list where each element is a list representing a cycle of
        alternatives that violate transitivity.

    Returns
    -------
    list of list
        A list where each element is a list containing a single formatted
        string representing the cycle in "A>B>C>A" format, clearly showing
        the circular preference relationship.

    Notes
    -----
    The formatting transforms cycles like ['A', 'B', 'C'] into strings
    like "A>B>C>A" to make transitivity violations more readable. The
    ">" symbol represents "is preferred to" or "dominates".

    A transitivity violation occurs when we have a cycle like:
    - A is preferred to B
    - B is preferred to C
    - C is preferred to A

    This creates a logical inconsistency that violates the transitivity
    property of rational preferences.

    Each formatted cycle is wrapped in a list to maintain consistency
    with other formatting functions and to allow for potential future
    extensions that might include additional metadata per cycle.

    Examples
    --------
    >>> cycles = [['A', 'B', 'C'], ['X', 'Y', 'Z', 'W']]
    >>> formatted = _format_transitivity_cycles(cycles)
    >>> print(formatted)
    [['A>B>C>A'], ['X>Y>Z>W>X']]
    >>>
    >>> # Each cycle shows the complete circular preference:
    >>> # First cycle: A dominates B, B dominates C, C dominates A
    >>> # Second cycle: X>Y>Z>W>X (4-way cycle)
    """
    result = []
    for subcycle in cycles:
        transformed = f">{subcycle}>{subcycle[0]}"
        result.append([transformed])
    return result


# =============================================================================
# CLASS
# =============================================================================


class TransitivityChecker(SKCMethodABC):
    """Robustness evaluator of an MCDM method.

    This checker verifies whether a method produces logically consistent and
    stable rankings when the original decision problem is decomposed into all
    possible pairs of alternatives.

    The evaluation is performed in two stages:

    1. **Transitivity Validation**:
    Check if the rankings derived from all two-alternative sub-problems follow
    the transitivity property.

    2. **Ranking Recomposition Consistency**:
    The criterion attempts to reconstruct a global ranking by combining the
    individual two-alternative rankings, offering various heuristics in case
    the sub-problems don't follow the transitivity property. This reconstructed
    ranking is then offered for comparrison as a `RanksComparator` for further
    analysis.

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
    """

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = [
        "dmaker",
        "random_state",
        "make_transitive_strategy",
        "max_ranks",
        "parallel_backend",
        "n_jobs",
    ]

    def __init__(
        self,
        dmaker,
        *,
        random_state=None,
        make_transitive_strategy="random",
        max_ranks=50,
        parallel_backend=None,
        n_jobs=None,
    ):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        # PARALLEL BACKEND
        self._parallel_backend = parallel_backend
        self._n_jobs = n_jobs

        # RANDOM
        self._random_state = np.random.default_rng(random_state)

        # STRATEGY FOR REMOVAL OF BREAKS IN TRANSITIVITY
        mk_transitive = CYCLE_REMOVAL_STRATEGIES.get(
            make_transitive_strategy, make_transitive_strategy
        )
        if not callable(mk_transitive):
            available_strategies = list(CYCLE_REMOVAL_STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy: {make_transitive_strategy}. \
                Available strategies: {available_strategies}"
            )
        self._make_transitive_strategy = mk_transitive

        # MAXIMIMUM PERMITED RANKS TO BE GENERATED
        self._max_ranks = int(max_ranks)  # TODO VER CONDICIONN

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dm = repr(self.dmaker)
        trs = self._make_transitive_strategy
        mr = self._max_ranks
        return (
            f"<{name} {dm}, "
            f"make_transitive_strategy={trs}, max_ranks={mr}>"
        )

    # PROPERTIES ==============================================================

    @property
    def dmaker(self):
        """The MCDA method, or pipeline to evaluate."""
        return self._dmaker

    @property
    def parallel_backend(self):
        """The parallel backend used \
        to generate all the alternatives."""
        return self._parallel_backend

    @property
    def random_state(self):
        """Controls the random state to generate variations in the \
        suboptimal alternatives."""
        return self._random_state

    @property
    def make_transitive_strategy(self):
        """The untie Strategy."""
        return self._make_transitive_strategy

    @property
    def max_ranks(self):
        """Maximum number of rankings to \
        be generated."""
        return self._max_ranks

    @property
    def n_jobs(self):
        """The number of parallel jobs used \
        in the generation."""
        return self._n_jobs

    # LOGIC ===================================================================

    def _evaluate_pairwise_submatrix(self, decision_matrix, alternative_pair):
        """Apply the MCDM pipeline to a sub-problem of two alternatives."""
        sub_dm = decision_matrix.loc[alternative_pair]
        return self._dmaker.evaluate(sub_dm)

    def _get_graph_edges(self, results, decision_matrix):
        """
        Generate directed graph edges from pairwise comparison results.

        Parameters
        ----------
        results : iterable
            Collection of comparison result objects. Each result must contain:
            - alternatives : list or tuple
                Names/identifiers of the two compared alternatives
            - rank_ : list or array-like
                Ranking values for each alternative \
                (lower values indicate better ranking)

        Returns
        -------
        list
            List of tuples (winner, loser) representing directed edges in the
            preference graph. Each tuple indicates that the first alternative
            is preferred over the second. For tied rankings, applies
            tie-breaking logic via dominance.

        Notes
        -----
        - Uses lower-is-better ranking system (rank 1 > rank 2 > rank 3)
        - Automatically handles tied rankings through internal tie-breaking \
            mechanism
        - Output format is suitable for constructing preference graphs
        """
        edges = []

        # Get the rank untier strategy
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
                untied_edges = _untie_by_dominance(
                    alt_names[0], alt_names[1], decision_matrix
                )
                edges.extend(untied_edges)

        return edges

    def _add_break_info_to_rank(self, rank, dag, removed_edges, iteration=1):
        """
        Add cycle-breaking information to a ranking result.

        This method enriches a RankResult object with information about the
        cycle-breaking process performed during graph decomposition, including
        the acyclic graph obtained and the edges that were removed to break
        cycles.

        Parameters
        ----------
        rank : RankResult
            The original ranking result to be enhanced with break information.
        dag : networkx.DiGraph
            The directed acyclic graph obtained after removing cycles from the
            original graph. This represents the acyclic version of the input
            graph.
        removed_edges : array-like
            Collection of edges that were removed from the original graph to
            break cycles and obtain the DAG.
        iteration : int, default 1
            The iteration number of the RRT3 recomposition process. Used to
            track multiple iterations of the cycle-breaking algorithm.

        Returns
        -------
        RankResult
            A new RankResult object containing the original ranking data plus
            additional information about the cycle-breaking process. The
            returned object includes:

            - Updated method name indicating RRT3 recomposition
            - Original alternatives and values preserved
            - Extended extra information with 'rrt23' entry containing:
                - acyclic_graph: the resulting DAG
                - removed_edges: edges removed during cycle breaking
        """
        method = rank.method
        if dag:
            method = f"{method} + RRT3 RECOMPOSITION_{iteration}"

        extra = dict(rank.extra_.items())

        extra["rrt23"] = Bunch(
            "rrt23",
            {
                "acyclic_graph": dag,
                "removed_edges": removed_edges,
            },
        )

        patched_rank = RankResult(
            method=method,
            alternatives=rank.alternatives,
            values=rank.values,
            extra=extra,
        )
        return patched_rank

    def _create_rank_from_dag(self, orank, dag, removed_edges, iteration=1):
        """
        Create a new ranking result from a directed acyclic graph (DAG).

        This method generates a new RankResult by computing rankings based on
        the in-degree sorting of a DAG. The ranking values are calculated using
        the topological order of nodes in the acyclic graph, and the result
        includes information about the cycle-breaking process.

        Parameters
        ----------
        orank : RankResult
            The original ranking result that serves as the base for creating
            the new ranking. Provides the method name, alternatives list, and
            extra information to be preserved in the new result.
        dag : networkx.DiGraph
            The directed acyclic graph from which to compute the new rankings.
            The graph should be acyclic and represent the structure after
            cycle-breaking.
        removed_edges : list or array-like
            Collection of edges that were removed from the original graph to
            create the DAG. This information is stored in the result
            for traceability.
        iteration : int, default 1
            The iteration number of the ranking process. Used to track multiple
            iterations of the cycle-breaking and ranking algorithm.

        Returns
        -------
        RankResult
            A new RankResult object containing:

            - method: Original method name from orank
            - alternatives: Same alternatives as in orank
            - values: New ranking values computed from DAG in-degree sorting
            - extra: Enhanced extra information
        """
        alternative_rank_value = _assign_rankings(_in_degree_sort(dag))

        rank = RankResult(
            method=orank.method,
            alternatives=orank.alternatives,
            values=np.array(
                [alternative_rank_value[alt] for alt in orank.alternatives]
            ),
            extra=orank.extra_,
        )

        rank = self._add_break_info_to_rank(
            rank, dag, removed_edges, iteration
        )

        return rank

    def _get_ranks(self, graph, orank):
        """
        Generate ranking results from a graph.

        This method produces one or more ranking results based on the input
        graph structure. If the graph is already acyclic, it generates a single
        ranking. If the graph contains cycles, it generates multiple rankings
        by creating different acyclic decompositions of the original graph.

        Parameters
        ----------
        graph : networkx.DiGraph
            The input graph from which to generate rankings. Can be either
            acyclic (DAG) or contain cycles.
        orank : RankResult
            The original ranking result that serves as the template for
            creating new rankings. Provides method name, alternatives, and
            extra information to be preserved across all generated rankings.

        Returns
        -------
        list of RankResult
            A list containing one or more RankResult objects:

            - If input graph is acyclic: Single RankResult with rankings based
                on the graph's topological structure
            - If input graph has cycles: Multiple RankResult objects, each
                corresponding to a different acyclic decomposition of the
                original graph
        """
        ranks = []

        if nx.is_directed_acyclic_graph(graph):
            rank = self._create_rank_from_dag(orank, graph, removed_edges=None)
            ranks.append(rank)

        else:
            acyclic_graphs = generate_acyclic_graphs(
                graph,
                strategy=self._make_transitive_strategy,
                max_graphs=self._max_ranks,
                seed=self._random_state,
            )

            for iteration, (dag, removed_edges) in enumerate(acyclic_graphs):
                rank = self._create_rank_from_dag(
                    orank, dag, removed_edges, iteration + 1
                )
                ranks.append(rank)

        return list(ranks)

    def _dominance_graph(self, dm, orank):
        """
        Create a directed dominance graph from pairwise alternative \
            comparisons.

        This method constructs a directed graph where nodes represent
        alternatives and edges represent dominance relationships. The graph is
        built by evaluating all pairwise combinations of alternatives using
        parallel processing for computational efficiency.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix containing alternatives and criteria values
            used for pairwise comparisons.
        orank : RankResult
            The original ranking result containing the list of alternatives to
            be compared pairwise.

        Returns
        -------
        networkx.DiGraph
            A directed graph where:
            - Nodes represent alternatives from orank.alternatives
            - Edges represent dominance relationships
                (A -> B means A dominates B)
            - All alternatives are guaranteed to be present as nodes, even if
                isolated
        """
        # Generate all pairwise combinations of alternatives
        pairwise_combinations = map(
            list, it.combinations(orank.alternatives, 2)
        )

        # Parallel processing of all pairwise sub-matrices
        # Each resulting sub-matrix has 2 alternatives × k original criteria
        with joblib.Parallel(
            prefer=self._parallel_backend, n_jobs=self._n_jobs
        ) as P:
            delayed_evaluation = joblib.delayed(
                self._evaluate_pairwise_submatrix
            )
            results = P(
                delayed_evaluation(dm, pair) for pair in pairwise_combinations
            )

        edges = self._get_graph_edges(results, dm)

        # Create directed graph
        graph = nx.DiGraph(edges)

        return graph

    def _calculate_transitivity_break(self, graph):
        """
        Calculate transitivity violations and their rate in a dominance graph.

        This method identifies cycles of length 3 (triangular cycles) in the
        graph, which represent violations of transitivity in preference
        relationships. A transitivity break occurs when A dominates B,
        B dominates C, but C dominates A.

        Parameters
        ----------
        graph : networkx.DiGraph
            The directed dominance graph to analyze for transitivity
            violations.

        Returns
        -------
        trans_break : list
            A formatted list of transitivity cycles found in the graph. Each
            cycle represents a violation of the transitivity property.
        trans_break_rate : float
            The rate of transitivity violations, calculated as the ratio of
            actual cycles to the theoretical maximum number of possible cycles
            for a graph with the given number of nodes.
        """
        trans_break = list(nx.simple_cycles(graph, length_bound=3))

        trans_break = _format_transitivity_cycles(trans_break)

        trans_break_rate = len(trans_break) / _transitivity_break_bound(
            len(graph.nodes)
        )

        return trans_break, trans_break_rate

    def _generate_graph_data(self, dm, orank):
        """
        Generate dominance graph and calculate transitivity metrics.

        This method combines the creation of a pairwise dominance graph with
        the calculation of transitivity break metrics, providing a
        comprehensive analysis of the decision problem's structure.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix containing alternatives and criteria for
            analysis.
        orank : RankResult
            The original ranking result containing alternatives to be analyzed.

        Returns
        -------
        graph : networkx.DiGraph
            The directed dominance graph representing pairwise relationships
            between alternatives.
        trans_break : list
            List of transitivity cycles (violations) found in the graph.
        trans_break_rate : float
            Normalized rate of transitivity violations
            (0.0 = perfect transitivity).
        """
        # Create pairwise dominance graph
        graph = self._dominance_graph(dm, orank)

        # Calculate transitivity break, and it's rate
        trans_break, trans_break_rate = self._calculate_transitivity_break(
            graph
        )
        return graph, trans_break, trans_break_rate

    def _test_criterion_2(self, trans_break_rate):
        """
        Perform test criterion 2: transitivity consistency check.

        This method evaluates whether the decision problem satisfies perfect
        transitivity by checking if the transitivity break rate is zero.

        Parameters
        ----------
        trans_break_rate : float
            The rate of transitivity violations in the dominance graph.
            Should be 0.0 for perfect transitivity.

        Returns
        -------
        str
            Test result status:
            - "Passed": No transitivity violations (trans_break_rate == 0)
            - "Not Passed": Transitivity violations detected
                (trans_break_rate > 0)
        """
        return "Passed" if trans_break_rate == 0 else "Not Passed"

    def _test_criterion_3(self, test_criterion_2, orank, returned_ranks):
        """
        Perform test criterion 3: ranking stability check.

        This method evaluates whether the ranking remains stable when the graph
        is perfectly transitive. It checks if the original ranking values match
        the first recomposed ranking when no transitivity violations exist.

        Parameters
        ----------
        test_criterion_2 : str
            Result of test criterion 2 ("Passed" or "Not Passed").
            Must be "Passed" for this test to potentially pass.
        orank : RankResult
            The original ranking result with baseline ranking values.
        returned_ranks : list of RankResult
            List of ranking results from graph recomposition. The first element
            is compared against the original ranking.

        Returns
        -------
        str
            Test result status:
            - "Passed": Test criterion 2 passed AND original ranking equals
            first recomposed ranking
            - "Not Passed": Either test criterion 2 failed OR rankings differ
        """
        return (
            "Passed"
            if (
                test_criterion_2 == "Passed"
                and (orank.values == returned_ranks[0].values).all()
            )
            else "Not Passed"
        )

    def evaluate(self, *, dm):
        """
        Execute the complete transitivity test and ranking analysis.

        This method performs a comprehensive transitivity analysis of a
        decision matrix, including dominance graph construction, transitivity
        testing, and ranking recomposition. It provides multiple ranking
        perspectives when cycles are present and diagnostic information about
        the decision problem's structure.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix to be evaluated, containing alternatives and
            criteria values for multi-criteria decision analysis.

        Returns
        -------
        RanksComparator
            A comprehensive result object containing:

            - Multiple named rankings (original + recompositions)
            - Diagnostic information in the `extra` attribute:
                - test_criterion_2: Transitivity consistency test result
                - test_criterion_3: Ranking stability test result
                - pairwise_dominance_graph: The constructed dominance graph
                - transitivity_break: List of transitivity violations
                - transitivity_break_rate: Normalized violation rate
        """
        dmaker = self._dmaker

        orank = dmaker.evaluate(dm)

        # add epmty info to orank
        orank = self._add_break_info_to_rank(
            orank, dag=None, removed_edges=None
        )

        # make the pairwise dominance graph and calculate transitivity metrics
        graph, trans_break, trans_break_rate = self._generate_graph_data(
            dm, orank
        )

        test_criterion_2 = self._test_criterion_2(trans_break_rate)

        # get the ranks from the graph
        returned_ranks = self._get_ranks(graph, orank)

        test_criterion_3 = self._test_criterion_3(
            test_criterion_2, orank, returned_ranks
        )

        names = ["Original"] + [
            f"Recomposition{i+1}" for i in range(len(returned_ranks))
        ]

        named_ranks = unique_names(
            names=names, elements=[orank] + returned_ranks
        )

        return RanksComparator(
            named_ranks,
            extra={
                "test_criterion_2": test_criterion_2,
                "pairwise_dominance_graph": graph,
                "test_criterion_3": test_criterion_3,
                "transitivity_break": trans_break,
                "transitivity_break_rate": trans_break_rate,
            },
        )
