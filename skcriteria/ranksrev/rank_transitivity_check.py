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
Transitivity Checker for MCDM Robustness Evaluation.

This module evaluates the logical consistency and stability of Multi-Criteria
Decision Making (MCDM) methods through transitivity analysis. It decomposes
decision problems into pairwise comparisons and reconstructs global rankings
to assess method robustness.

The module validates whether rankings satisfy the transitivity property
(if A ≻ B and B ≻ C, then A ≻ C) and provides mechanisms to handle violations.

Key Features
------------
- Transitivity validation through pairwise decomposition
- Ranking recomposition with cycle-breaking strategies
- Comprehensive diagnostic reporting

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import itertools as it

    import joblib

    import networkx as nx

    import numpy as np

    from ..agg import RankResult
    from ..cmp import RanksComparator
    from ..core import SKCMethodABC
    from ..utils import Bunch, unique_names
    from ..utils.cycle_removal import (
        CYCLE_REMOVAL_STRATEGIES,
        generate_acyclic_graphs,
    )
    from ..tiebreaker import FallbackTieBreaker


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


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
    return n * (n**2 - 4) // 24 if n % 2 == 0 else n * (n**2 - 1) // 24


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


class RankTransitivityChecker(SKCMethodABC):
    """
    Robustness evaluator for Multi-Criteria Decision Making (MCDM) methods.

    This class validates the logical consistency and stability of MCDM method
    rankings by analyzing transitivity properties through pairwise alternative
    comparisons.
    It identifies ranking inconsistencies and provides alternative ranking
    reconstructions when transitivity violations occur.

    The evaluation process is the following:

    1. **Pairwise Dominance Analysis**:
       Evaluates all possible pairs of alternatives using the provided MCDM
       method to construct a directed dominance graph representing preference
       relationships.

    2. **Transitivity Validation** (Test Criterion 2):
       Detects cycles in the dominance graph that violate the transitivity
       property. A transitive ranking requires that if A > B and B > C, then
       A > C must hold.

    3. **Ranking Stability Assessment** (Test Criterion 3):
       Compares the original ranking with reconstructed rankings to evaluate
       consistency when the decision problem is decomposed and recomposed.

    4. **Ranking Reconstruction**:
       When transitivity violations exist, applies cycle-breaking strategies to
       generate alternative valid rankings through graph decomposition
       techniques.

    Parameters
    ----------
    dmaker : object
        Decision maker instance that must implement an ``evaluate(dm)`` method.
        This represents the MCDM method or pipeline to be evaluated for
        robustness.

    fallback : object
        Optional fallback decision maker for tie-breaking in pairwise
        comparisons. Must also implement an ``evaluate(dm)`` method.
        If not provided, lexicographical tie breaking is used.

    random_state : int, numpy.random.Generator, or None, default=None
        Controls randomization in cycle-breaking strategies and alternative
        ranking generation. Ensures reproducible results when set to a
        specific integer.

    allow_missing_alternatives : bool, default=False
        Whether to allow rankings that don't include all original alternatives
        (using a pipeline that implements a filter, for example can remove
        alternatives).
        When False, raises ValueError if any alternative is missing from
        results. When True, missing alternatives are assigned the worst
        ranking + 1.

    cycle_removal_strategy : str or callable, default="random"
        Strategy for breaking cycles in non-transitive dominance graphs.
        Available built-in strategies include cycle removal heuristics.
        Can also accept custom callable functions for specialized approaches.

    max_ranks : int, default=50
        Maximum number of alternative rankings to generate when breaking
        cycles. Controls computational complexity by limiting the number of
        decompositions.

    parallel_backend : str or None, default=None
        Backend for parallel computation of pairwise evaluations.
        Options include 'threading', 'multiprocessing', or None for sequential.
        Improves performance for large numbers of alternatives.

    n_jobs : int or None, default=None
        Number of parallel jobs for pairwise evaluation. When None, uses all
        available processors. Set to 1 for sequential processing.

    Raises
    ------
    TypeError
        If ``dmaker`` doesn't implement the required ``evaluate()`` method.

    ValueError
        If ``cycle_removal_strategy`` is not a valid strategy name or \
            callable.
        If ``allow_missing_alternatives=False`` and alternatives are missing \
            from results.

    Examples
    --------
    Basic usage with an MCDM method:

    >>> from skcriteria.preprocessing import invert_objectives
    >>> from skcriteria.agg import simple
    >>>
    >>> # Create a decision maker
    >>> dm_method = simple.WeightedSum()
    >>>
    >>> # Initialize transitivity checker
    >>> checker = RankTransitivityChecker(dm_method)
    >>>
    >>> # Evaluate a decision matrix
    >>> result = checker.evaluate(dm=decision_matrix)
    >>>
    >>> # Check test results
    >>> print(f"Test Criterion 2: {result.extra['test_criterion_2']}")
    >>> print(f"Test Criterion 3: {result.extra['test_criterion_3']}")

    Advanced configuration with custom parameters:

    >>> checker = RankTransitivityChecker(
    ...     dmaker=dm_method,
    ...     random_state=42,
    ...     allow_missing_alternatives=True,
    ...     cycle_removal_strategy="random",
    ...     max_ranks=100,
    ...     parallel_backend="threading",
    ...     n_jobs=4
    ... )
    """

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = [
        "dmaker",
        "fallback",
        "random_state",
        "allow_missing_alternatives",
        "cycle_removal_strategy",
        "max_ranks",
        "parallel_backend",
        "n_jobs",
    ]

    def __init__(
        self,
        dmaker,
        *,
        fallback=None,
        random_state=None,
        allow_missing_alternatives=False,
        cycle_removal_strategy="random",
        max_ranks=50,
        parallel_backend=None,
        n_jobs=None,
    ):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        if fallback:
            if not (
                hasattr(fallback, "evaluate") and callable(fallback.evaluate)
            ):
                raise TypeError(
                    "'fallback' must implement 'evaluate()' method"
                )

            self._pair_evaluator = FallbackTieBreaker(dmaker, fallback)

        else:
            self._pair_evaluator = dmaker

        self._fallback = fallback

        # ALLOW MISSING ALTERNATIVES
        self._allow_missing_alternatives = bool(allow_missing_alternatives)

        # PARALLEL BACKEND
        self._parallel_backend = parallel_backend
        self._n_jobs = n_jobs

        # RANDOM
        self._random_state = np.random.default_rng(random_state)

        # STRATEGY FOR REMOVAL OF BREAKS IN TRANSITIVITY
        mk_transitive = CYCLE_REMOVAL_STRATEGIES.get(
            cycle_removal_strategy, cycle_removal_strategy
        )
        if not callable(mk_transitive):
            available_strategies = list(CYCLE_REMOVAL_STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy: {cycle_removal_strategy}. \
                Available strategies: {available_strategies}"
            )
        self._cycle_removal_strategy = mk_transitive

        # MAXIMIMUM PERMITED RANKS TO BE GENERATED
        if max_ranks < 1:
            raise ValueError(
                f"max_ranks should be greater than zero, current \
                    value {max_ranks}"
            )
        self._max_ranks = int(max_ranks)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dm = repr(self.dmaker)
        trs = self._cycle_removal_strategy
        mr = self._max_ranks
        return (
            f"<{name} {dm}, " f"cycle_removal_strategy={trs}, max_ranks={mr}>"
        )

    # PROPERTIES ==============================================================

    @property
    def dmaker(self):
        """The MCDA method, or pipeline to evaluate."""
        return self._dmaker

    @property
    def fallback(self):
        """The MCDA method, or pipeline to evaluate for tie breaking."""
        return self._fallback

    @property
    def random_state(self):
        """Controls the random state to generate variations in the \
        suboptimal alternatives."""
        return self._random_state

    @property
    def allow_missing_alternatives(self):
        """Whether rankings are allowed that don't contain all original \
        alternatives."""
        return self._allow_missing_alternatives

    @property
    def cycle_removal_strategy(self):
        """The strategy function used for breaking transitivity cycles."""
        return self._cycle_removal_strategy

    @property
    def max_ranks(self):
        """Maximum number of rankings to be generated."""
        return self._max_ranks

    @property
    def parallel_backend(self):
        """The parallel backend used to generate all the alternatives."""
        return self._parallel_backend

    @property
    def n_jobs(self):
        """The number of parallel jobs used in the pairwise evaluations."""
        return self._n_jobs

    # LOGIC ===================================================================

    def _evaluate_pairwise_submatrix(self, decision_matrix, alternative_pair):
        """
        Apply the MCDM pipeline to a sub-problem of two alternatives.

        This method extracts a submatrix containing only the specified pair of
        alternatives from the decision matrix and evaluates it using the
        configured decision maker.

        Parameters
        ----------
        decision_matrix : pandas.DataFrame
            The complete decision matrix with alternatives as rows and criteria
            as columns. Must contain the alternatives specified in
            alternative_pair.
        alternative_pair : list, tuple, or array-like
            Collection of exactly two alternative identifiers/names that exist
            as row indices in the decision_matrix. These alternatives will be
            extracted for pairwise comparison.

        Returns
        -------
        RankResult
            The result of applying the MCDM evaluation method to the submatrix
            containing only the two specified alternatives. The exact type and
            structure depends on the specific decision maker (self._dmaker)
            being used.

        Notes
        -----
        This method is typically used internally for pairwise comparison
        approaches in multi-criteria decision making, where the overall
        problem is decomposed into smaller two-alternative subproblems.
        """
        sub_dm = decision_matrix.loc[alternative_pair]
        return self._pair_evaluator.evaluate(sub_dm)

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
            else:
                edges.append((alt_names[1], alt_names[0]))

        return edges

    def _add_break_info_to_rank(
        self, rank, dag, removed_edges, full_alternatives, iteration
    ):
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
            - Extended extra information with 'transitivity_check' entry \
                containing:
                - acyclic_graph: the resulting DAG
                - removed_edges: edges removed during cycle breaking
        """
        alternatives = rank.alternatives
        values = rank.values
        method = rank.method
        if dag:
            method = f"{method} + RRT3 RECOMPOSITION_{iteration}"

        # we check if the decision_maker did not eliminate any alternatives
        alts_diff = np.setxor1d(alternatives, full_alternatives)
        has_missing_alternatives = len(alts_diff) > 0

        if has_missing_alternatives:
            # if a missing alternative are not allowed must raise an error
            if not self._allow_missing_alternatives:
                raise ValueError(f"Missing alternative/s {set(alts_diff)!r}")

            # add missing alternatives with the  worst ranking + 1
            fill_values = np.full_like(alts_diff, rank.rank_.max() + 1)

            # concatenate the missing alternatives and the new rankings
            alternatives = np.concatenate((alternatives, alts_diff))
            values = np.concatenate((values, fill_values))

        extra = dict(rank.extra_.items())

        extra["transitivity_check"] = Bunch(
            "transitivity_check",
            {
                "acyclic_graph": dag,
                "removed_edges": removed_edges,
                "missing_alternatives": alts_diff,
            },
        )

        return RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def _create_rank_from_dag(
        self, rrank, dag, removed_edges, full_alternatives, iteration=1
    ):
        """
        Create a new ranking result from a directed acyclic graph (DAG).

        This method generates a new RankResult by computing rankings based on
        the in-degree sorting of a DAG. The ranking values are calculated using
        the topological order of nodes in the acyclic graph, and the result
        includes information about the cycle-breaking process.

        Parameters
        ----------
        rrank : RankResult
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

            - method: Original method name from rrank
            - alternatives: Same alternatives as in rrank
            - values: New ranking values computed from DAG in-degree sorting
            - extra: Enhanced extra information
        """
        alternative_rank_value = _assign_rankings(_in_degree_sort(dag))

        rank = RankResult(
            method=rrank.method,
            alternatives=rrank.alternatives,
            values=np.array(
                [alternative_rank_value[alt] for alt in rrank.alternatives]
            ),
            extra=rrank.extra_,
        )

        rank = self._add_break_info_to_rank(
            rank, dag, removed_edges, full_alternatives, iteration
        )

        return rank

    def _generate_reconstructed_ranks(self, graph, rrank, full_alternatives):
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
        rrank : RankResult
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
            rank = self._create_rank_from_dag(
                rrank,
                graph,
                removed_edges=None,
                full_alternatives=full_alternatives,
            )
            ranks.append(rank)

            return ranks

        acyclic_graphs = generate_acyclic_graphs(
            graph,
            strategy=self._cycle_removal_strategy,
            max_graphs=self._max_ranks,
            seed=self._random_state,
        )

        for iteration, (dag, removed_edges) in enumerate(acyclic_graphs):
            rank = self._create_rank_from_dag(
                rrank, dag, removed_edges, full_alternatives, iteration + 1
            )
            ranks.append(rank)

        return ranks

    def _dominance_graph(self, dm, rrank):
        """
        Create a directed dominance graph from pairwise alternative \
            comparisons.

        This method constructs a directed graph where nodes represent
        alternatives and edges represent dominance relationships. The graph is
        built by evaluating all pairwise combinations of alternatives.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix containing alternatives and criteria values
            used for pairwise comparisons.
        rrank : RankResult
            The original ranking result containing the list of alternatives to
            be compared pairwise.

        Returns
        -------
        networkx.DiGraph
            A directed graph where:
            - Nodes represent alternatives from rrank.alternatives
            - Edges represent dominance relationships
                (A -> B means A dominates B)
            - All alternatives are guaranteed to be present as nodes, even if
                isolated
        """
        # Generate all pairwise combinations of alternatives
        pairwise_combinations = map(
            list, it.combinations(rrank.alternatives, 2)
        )

        # Parallel processing of all pairwise sub-matrices
        # Each resulting sub-matrix has 2 alternatives × k original criteria
        # TODO: Probar sacar paralelismo
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

    def _generate_graph_data(self, dm, rrank):
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
        rrank : RankResult
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
        graph = self._dominance_graph(dm, rrank)

        # Calculate transitivity break, and it's rate
        trans_break, trans_break_rate = self._calculate_transitivity_break(
            graph
        )

        return graph, trans_break, trans_break_rate

    def _test_criterion_2(self, dm, orank):
        """
        Perform test criterion 2: transitivity consistency check.

        This method evaluates whether the decision problem satisfies perfect
        transitivity. It generates a pairwise dominance graph and calculates
        transitivity metrics to assess the consistency of the MCDM.

        Parameters
        ----------
        dm : array-like
            Decision matrix containing the alternatives and criteria values.
        orank : array-like
            Ranking or ordering information for the alternatives.

        Returns
        -------
        tuple
            A tuple containing four elements:
            - graph : object
                The pairwise dominance graph structure.
            - trans_break : int or float
                The absolute number of transitivity violations detected.
            - trans_break_rate : float
                The rate of transitivity violations in the dominance graph.
                Value of 0.0 indicates perfect transitivity.
            - test_criterion_2 : boolean
                Test result status:
                - True: No transitivity violations (trans_break_rate == 0)
                - False: Transitivity violations detected
                (trans_break_rate > 0)

        Notes
        -----
        This test is crucial for validating the logical consistency of decision
        rankings. Perfect transitivity means that if alternative A dominates B
        and B dominates C, then A must also dominate C.
        """
        # make the pairwise dominance graph and calculate transitivity metrics
        graph, trans_break, trans_break_rate = self._generate_graph_data(
            dm, orank
        )

        test_criterion_2 = trans_break_rate == 0
        return test_criterion_2, graph, trans_break, trans_break_rate

    def _test_criterion_3(self, test_criterion_2, rrank, returned_ranks):
        """
        Perform test criterion 3: ranking stability check.

        Parameters
        ----------
        test_criterion_2 : bool
            Result of test criterion 2.
            Must be True for this test to potentially pass.
        rrank : RankResult
            The original ranking result with baseline ranking values.
        returned_ranks : list of RankResult
            List of ranking results from graph recomposition. The first element
            is compared against the original ranking.

        Returns
        -------
        bool
            Test result status:
            - True: Test criterion 2 passed AND original ranking equals
            first recomposed ranking
            - False: Either test criterion 2 failed OR rankings differ
        """
        return (
            test_criterion_2
            and (rrank.values == returned_ranks[0].values).all()
        )

    def evaluate(self, *, dm):
        """
        Execute the complete transitivity test and ranking analysis.

        This method performs a comprehensive transitivity analysis,
        including dominance graph construction, transitivity testing, and
        ranking recomposition. It provides multiple ranking perspectives when
        cycles are present and diagnostic information about the decision
        problem's structure.

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
        full_alternatives = dm.alternatives

        # we need a first reference ranking
        rrank = dmaker.evaluate(dm)
        patched_rrank = self._add_break_info_to_rank(
            rrank,
            dag=None,
            removed_edges=None,
            full_alternatives=full_alternatives,
            iteration=None,
        )

        # make the pairwise dominance graph and calculate transitivity metrics
        test_criterion_2, graph, trans_break, trans_break_rate = (
            self._test_criterion_2(dm, rrank)
        )

        # get the ranks from the graph
        reconstructed_ranks = self._generate_reconstructed_ranks(
            graph, rrank, full_alternatives
        )

        test_criterion_3 = self._test_criterion_3(
            test_criterion_2, patched_rrank, reconstructed_ranks
        )

        names = ["Original"] + [
            f"Recomposition{i+1}" for i in range(len(reconstructed_ranks))
        ]

        named_ranks = unique_names(
            names=names, elements=[patched_rrank] + reconstructed_ranks
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
