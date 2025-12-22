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
    from ..utils import Bunch, unique_names, deprecate
    from ..utils.cycle_removal import (
        CYCLE_REMOVAL_STRATEGIES,
    )
    from ..utils import dag_rank
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

    preferred_parallel_backend : str or None, default=None
        Backend for parallel computation of pairwise evaluations.
        Options include 'threading', 'multiprocessing', or None for sequential.
        Improves performance for large numbers of alternatives.

    n_jobs : int or None, default=None
        Number of parallel jobs for pairwise evaluation. When None, uses all
        available processors. Set to 1 for sequential processing.

    parallel_backend : str or None, default=None (deprecated)
        Use ``preferred_parallel_backend`` instead.

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
    [CLAUDE COMPLETE]
    """

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = [
        "dmaker",
        "fallback",
        "random_state",
        "allow_missing_alternatives",
        "max_ranks",
        "fas_method",
        "preferred_parallel_backend",
        "n_jobs",
    ]

    def __init__(
        self,
        dmaker,
        *,
        fallback=None,
        random_state=None,
        allow_missing_alternatives=False,
        max_ranks=50,
        fas_method="auto",
        preferred_parallel_backend=None,
        n_jobs=None,
        parallel_backend=None,
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
        if (
            parallel_backend is not None
            and preferred_parallel_backend is not None
        ):
            raise ValueError(
                "Only one of 'parallel_backend' (deprecated since 0.9.1) and "
                "'preferred_parallel_backend' can be specified"
            )
        if parallel_backend is not None:
            deprecate.warn(
                "The 'parallel_backend' parameter is deprecated since 0.9.1,  "
                "use 'preferred_parallel_backend' instead."
            )
            preferred_parallel_backend = parallel_backend

        self._preferred_parallel_backend = preferred_parallel_backend
        self._n_jobs = None if n_jobs is None else int(n_jobs)

        # RANDOM
        self._random_state = np.random.default_rng(random_state)

        # MAXIMIMUM PERMITED RANKS TO BE GENERATED
        if max_ranks < 1:
            raise ValueError(
                f"max_ranks should be greater than zero, current \
                    value {max_ranks}"
            )
        self._max_ranks = int(max_ranks)

        # FAS METHOD
        self._fas_method = fas_method

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dm = repr(self.dmaker)
        fm = self._fas_method
        mr = self._max_ranks
        return (
            f"<{name} {dm}, " f"fas_method={fm}, max_ranks={mr}>"
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
    def max_ranks(self):
        """Maximum number of rankings to be generated."""
        return self._max_ranks

    @property
    def fas_method(self):
        """[CLAUDE COMPLETE]"""
        return self._fas_method

    @property
    def preferred_parallel_backend(self):
        """The parallel backend used to generate all the alternatives."""
        return self._preferred_parallel_backend

    @property
    @deprecate.deprecated(
        reason="Use 'preferred_parallel_backend' instead", version="0.9.1"
    )
    def parallel_backend(self):
        """The parallel backend used to generate all the alternatives."""
        return self.preferred_parallel_backend

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

    def _add_info_to_rank(self, rank, full_alternatives, recomposition_number=None):
        """
        Add transitivity check metadata to a ranking result.

        This method enriches a ranking result with additional information about
        missing alternatives and recomposition status, ensuring all alternatives
        from the original decision matrix are accounted for.

        Parameters
        ----------
        rank : RankResult
            The ranking result to be enriched with metadata.
        full_alternatives : array-like
            Complete array of all alternatives from the original decision matrix.
        recomposition_number : int, optional
            The recomposition iteration number. If None (default), this is the
            original ranking. If an integer, this is a reconstructed ranking from
            the DAG and the method name will be updated accordingly.

        Returns
        -------
        RankResult
            A new RankResult with updated method name (if recomposed), all
            alternatives included (missing ones get worst rank + 1), and
            transitivity check metadata in the extra attribute.

        Raises
        ------
        ValueError
            If allow_missing_alternatives is False and some alternatives are
            missing from the ranking.
        """
        alternatives = rank.alternatives
        values = rank.values
        method = rank.method
        if recomposition_number:
            method = f"{method} + RECOMPOSITION_{recomposition_number}"

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
                "missing_alternatives": alts_diff,
                "recomposition": recomposition_number,
            },
        )

        return RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def _reconstruct_ranks_from_dag(self, graph, rrank, full_alternatives):
        """
        Reconstruct rankings from a dominance graph using DAG conversion.

        This method converts the dominance graph to a DAG by removing feedback
        arcs, then generates all possible rankings from the DAG's topological
        sorts.

        Parameters
        ----------
        graph : networkx.DiGraph
            The dominance graph to convert to a DAG.
        rrank : RankResult
            The reference ranking result used as a template for reconstructed
            rankings.
        full_alternatives : array-like
            Array of all alternatives that should be included in the rankings.

        Returns
        -------
        ranks : list of RankResult
            List of reconstructed ranking results, one for each topological
            sort of the DAG (up to max_ranks limit).
        fas : list of tuple
            The feedback arc set (edges removed to make the graph acyclic).
        method : str or None
            The method used for feedback arc set computation.
        """
        dag, fas, method = dag_rank.as_dag(
            graph=graph, method=self._fas_method
        )

        all_rankings = dag_rank.all_rankings(
            full_alternatives, dag, self._max_ranks
        )

        ranks = []
        for rnumber, rank_values in enumerate(all_rankings):
            rank = RankResult(
                method=rrank.method,
                alternatives=rrank.alternatives,
                values=rank_values,
                extra=rrank.extra_,
            )
            rank = self._add_info_to_rank(rank, full_alternatives, rnumber)
            ranks.append(rank)

        return ranks, fas, method

    def _dominance_graph(self, dm, rrank):
        """
        Create a directed dominance graph from pairwise alternative comparisons.

        This method constructs a directed graph where nodes represent
        alternatives and edges represent dominance relationships. The graph is
        built by evaluating all pairwise combinations of alternatives using
        the configured parallel backend.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix containing alternatives and criteria values
            used for pairwise comparisons.
        rrank : RankResult
            The reference ranking result containing the list of alternatives to
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
        preferred_parallel_backend = self._preferred_parallel_backend
        n_jobs = self._n_jobs

        # Generate all pairwise combinations of alternatives
        pairwise_combinations = map(
            list, it.combinations(rrank.alternatives, 2)
        )

        # Parallel processing of all pairwise sub-matrices
        # Each resulting sub-matrix has 2 alternatives × k original criteria
        # TODO: Probar sacar paralelismo
        with joblib.Parallel(
            prefer=preferred_parallel_backend, n_jobs=n_jobs
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

    def _build_dominance_graph_with_metrics(
        self,
        dm,
        rrank,
    ):
        """
        Build dominance graph and calculate transitivity metrics.

        This method combines the creation of a pairwise dominance graph with
        the calculation of transitivity break metrics, providing a
        comprehensive analysis of the decision problem's structure.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix containing alternatives and criteria for
            analysis.
        rrank : RankResult
            The reference ranking result containing alternatives to be analyzed.

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

    def _check_transitivity_consistency(self, dm, rrank):
        """
        Check transitivity consistency (test criterion 2).

        This method evaluates whether the decision problem satisfies perfect
        transitivity. It builds a pairwise dominance graph and calculates
        transitivity metrics to assess the consistency of the MCDM method.

        Parameters
        ----------
        dm : DecisionMatrix
            Decision matrix containing the alternatives and criteria values.
        rrank : RankResult
            The reference ranking result containing the alternatives to be analyzed.

        Returns
        -------
        test_criterion_2 : bool
            Test result status:
            - True: No transitivity violations (trans_break_rate == 0)
            - False: Transitivity violations detected (trans_break_rate > 0)
        graph : networkx.DiGraph
            The pairwise dominance graph structure.
        trans_break : list
            List of transitivity cycles (violations) found in the graph.
        trans_break_rate : float
            The rate of transitivity violations in the dominance graph.
            Value of 0.0 indicates perfect transitivity.

        Notes
        -----
        This test is crucial for validating the logical consistency of decision
        rankings. Perfect transitivity means that if alternative A dominates B
        and B dominates C, then A must also dominate C.
        """
        # make the pairwise dominance graph and calculate transitivity metrics
        graph, trans_break, trans_break_rate = (
            self._build_dominance_graph_with_metrics(dm, rrank)
        )

        test_criterion_2 = trans_break_rate == 0
        return test_criterion_2, graph, trans_break, trans_break_rate

    def _check_ranking_stability(
        self, test_criterion_2, rrank, returned_ranks
    ):
        """
        Check ranking stability (test criterion 3).

        This method verifies that the reference ranking is stable by comparing
        it with the first reconstructed ranking from the DAG. The test only
        passes if transitivity is satisfied and rankings match.

        Parameters
        ----------
        test_criterion_2 : bool
            Result of transitivity consistency check.
            Must be True for this test to potentially pass.
        rrank : RankResult
            The reference ranking result with baseline ranking values.
        returned_ranks : list of RankResult
            List of ranking results from DAG reconstruction. The first element
            is compared against the reference ranking.

        Returns
        -------
        bool
            Test result status:
            - True: Transitivity check passed AND reference ranking equals
              first reconstructed ranking
            - False: Either transitivity check failed OR rankings differ
        """
        return (
            test_criterion_2
            and (rrank.values == returned_ranks[0].values).all()
        )

    def evaluate(self, dm):
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
        full_alternatives = np.array(dm.alternatives)

        # we need a first reference ranking
        rrank = dmaker.evaluate(dm)
        patched_rrank = self._add_info_to_rank(
            rrank, full_alternatives=full_alternatives
        )

        # make the pairwise dominance graph and calculate transitivity metrics
        # Test criterion 2
        test_criterion_2, graph, trans_break, trans_break_rate = (
            self._check_transitivity_consistency(dm, rrank)
        )

        # get the ranks from the graph
        reconstructed_ranks = self._reconstruct_ranks_from_dag(
            graph, rrank, full_alternatives
        )

        test_criterion_3 = self._check_ranking_stability(
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
