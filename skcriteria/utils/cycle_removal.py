#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utility to remove cycles from Networkx graphs."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import Counter

import networkx as nx

import numpy as np

# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================


def _cycle_to_edges(cycle):
    """
    Convert a cycle (list of nodes) to a list of edges.

    This utility function transforms a cycle represented as a sequence of nodes
    into a list of directed edges that form the cycle. The last node connects
    back to the first node to complete the cycle.

    Parameters
    ----------
    cycle : list
        A list of nodes representing a cycle in the graph. The nodes should
        be in the order they appear in the cycle.

    Returns
    -------
    list of tuple
        A list of tuples where each tuple (u, v) represents a directed edge
        from node u to node v in the cycle. The last edge connects the final
        node back to the first node.
    """
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _select_edge_random(cycle, edge_freq, rng):
    """
    Select a random edge from a cycle with uniform probability.

    This function implements a random edge selection strategy where each edge
    in the cycle has an equal probability of being selected for removal.

    Parameters
    ----------
    cycle : list
        A list of nodes representing a cycle in the graph.
    edge_freq : Counter
        Edge frequency counter (not used in random selection, but maintained
        for interface consistency).
    rng : numpy.random.Generator
        Random number generator for reproducible random selection.

    Returns
    -------
    tuple
        A tuple (u, v) representing the selected edge to be removed from
        the cycle.

    Notes
    -----
    This strategy provides unbiased edge selection, giving each edge in the
    cycle an equal chance of being removed. It's useful when no prior
    information about edge importance is available or when uniform sampling
    is desired.
    """
    edges = _cycle_to_edges(cycle)
    return tuple(rng.choice(edges))


def _select_edge_weighted(cycle, edge_freq, rng):
    """
    Select edge with probability proportional to frequency.

    This function implements a weighted edge selection strategy where edges
    that appear in more cycles are more likely to be selected for removal.
    The probability of selection is proportional to the edge's frequency
    across all cycles in the graph.

    Parameters
    ----------
    cycle : list
        A list of nodes representing a cycle in the graph.
    edge_freq : Counter
        Counter object containing the frequency of each edge across all
        cycles in the graph. Used to calculate selection probabilities.
    rng : numpy.random.Generator
        Random number generator for reproducible weighted selection.

    Returns
    -------
    tuple
        A tuple (u, v) representing the selected edge to be removed from
        the cycle.

    Notes
    -----
    This strategy prioritizes removal of edges that participate in many cycles,
    potentially leading to more efficient cycle breaking.

    The selection probability for edge e is:
    P(e) = (freq(e) + 1) / sum(freq(all_edges) + 1)
    """
    edges = _cycle_to_edges(cycle)
    weights = [edge_freq[edge] + 1 for edge in edges]
    return tuple(rng.choice(edges, p=np.array(weights) / np.sum(weights)))


# =============================================================================
# CONSTANTS
# =============================================================================


CYCLE_REMOVAL_STRATEGIES = {
    "random": _select_edge_random,
    "weighted": _select_edge_weighted,
}


# =============================================================================
# MAIN FUNCTIONALITY
# =============================================================================


def _calculate_edge_frequencies(graph):
    """
    Calculate edge frequencies across all cycles in the graph.

    This function computes how many times each edge appears in cycles
    throughout the graph. This information is used by weighted edge
    selection strategies to prioritize edges that participate in multiple
    cycles.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph for which to calculate edge frequencies.

    Returns
    -------
    Counter
        A Counter object where keys are edge tuples (u, v) and values
        are the number of cycles containing that edge.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (2, 4), (4, 3)])
    >>> freq = _calculate_edge_frequencies(G)
    >>> freq[(2, 3)]  # Edge (2,3) appears in multiple cycles
    2
    """
    edge_freq = Counter()
    for cycle in nx.simple_cycles(graph):
        edges = _cycle_to_edges(cycle)
        edge_freq.update(edges)
    return edge_freq


def generate_acyclic_graphs(
    graph, *, strategy="random", max_graphs=10, seed=None
):
    """
    Generate multiple acyclic graphs by removing edges from cycles.

    This function creates multiple directed acyclic graphs (DAGs) from a
    potentially cyclic input graph by strategically removing edges that
    participate in cycles. Different strategies can be used to select
    which edges to remove, and multiple attempts generate diverse solutions.

    Parameters
    ----------
    graph : networkx.DiGraph
        The input directed graph, which may contain cycles.
    strategy : str or callable, default "random"
        Edge selection strategy for cycle breaking:
        - "random": Select edges uniformly at random
        - "weighted": Select edges proportional to their cycle frequency
        - callable: Custom edge selection function with signature \
            func(cycle, edge_freq, rng) -> edge_tuple
    max_graphs : int, default 10
        Maximum number of acyclic graphs to generate. The function may
        return fewer graphs if it cannot generate enough unique solutions.
    seed : int, optional
        Random seed for reproducible results. If None, results will vary
        between runs.

    Returns
    -------
    list of tuple
        A list of tuples where each tuple contains:
        - acyclic_graph (networkx.DiGraph): A directed acyclic graph
        - removed_edges (set): Set of edges that were removed to break cycles

        If the input graph is already acyclic, returns a single tuple with
        the original graph and an empty set of removed edges.

    Raises
    ------
    ValueError
        If the strategy parameter is not recognized and not a callable
        function.

    Notes
    -----
    The algorithm works by:

    1. **Cycle Detection**: Find all simple cycles in the input graph
    2. **Edge Selection**: For each cycle, select an edge to remove based
       on the chosen strategy
    3. **Graph Modification**: Create a new graph with selected edges removed
    4. **Validation**: Check if the resulting graph is acyclic
    5. **Iteration**: Repeat with different random choices to generate
       multiple solutions

    The function attempts up to `2 * max_graphs` iterations to generate
    the requested number of acyclic graphs. This provides robustness
    against cases where the random selection produces duplicate solutions.

    **Strategy Details:**

    - **Random**: Each edge in each cycle has equal probability of removal
    - **Weighted**: Edges appearing in more cycles are more likely to be
        removed

    **Performance Considerations:**

    - Time complexity depends on the number of cycles and their lengths
    - Memory usage scales with the number of generated graphs
    - For large graphs with many cycles, consider reducing `max_graphs`

    Examples
    --------
    >>> import networkx as nx
    >>>
    >>> # Create a cyclic graph
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (1, 4)])
    >>>
    >>> # Generate acyclic graphs using random strategy
    >>> results = generate_acyclic_graphs(G, strategy="random", max_graphs=5)
    >>> len(results)
    5
    >>>
    >>> # Check that results are acyclic
    >>> for dag, removed in results:
    ...     print(f"Acyclic: {nx.is_directed_acyclic_graph(dag)}")
    ...     print(f"Removed edges: {removed}")
    Acyclic: True
    Removed edges: {(3, 1)}
    Acyclic: True
    Removed edges: {(2, 3)}
    ...

    >>> # Use weighted strategy for more systematic edge removal
    >>> results_weighted = generate_acyclic_graphs(
    ...     G, strategy="weighted", max_graphs=3, seed=42
    ... )
    >>>
    >>> # Already acyclic graph
    >>> dag = nx.DiGraph([(1, 2), (2, 3)])
    >>> results_acyclic = generate_acyclic_graphs(dag)
    >>> len(results_acyclic)
    1
    >>> results_acyclic[0][1]  # No edges removed
    set()

    See Also
    --------
    networkx.simple_cycles : Find all simple cycles in a directed graph
    networkx.is_directed_acyclic_graph : Check if a graph is acyclic
    """
    rng = np.random.default_rng(seed)
    acyclic_graphs = []
    cycles = list(nx.simple_cycles(graph))
    max_attempts = 2 * max_graphs

    # If the graph is already acyclic, return it with no edges removed
    if nx.is_directed_acyclic_graph(graph):
        return [(graph.copy(), set())]

    # Validate strategy
    select_edge = CYCLE_REMOVAL_STRATEGIES.get(strategy, strategy)
    if not callable(select_edge):
        available_strategies = list(CYCLE_REMOVAL_STRATEGIES.keys())
        raise ValueError(
            f"Unknown strategy: {strategy}. \
            Available strategies: {available_strategies}"
        )
    edge_freq = _calculate_edge_frequencies(graph)

    attempts = 0
    while attempts < max_attempts and len(acyclic_graphs) < max_graphs:
        attempts += 1
        to_remove = set()

        # Select edges to remove from each cycle
        for cycle in cycles:
            edge_to_remove = select_edge(cycle, edge_freq, rng)
            to_remove.add(edge_to_remove)

        # Creates acyclic graph
        modified_graph = graph.copy()
        modified_graph.remove_edges_from(to_remove)

        if nx.is_directed_acyclic_graph(modified_graph):
            acyclic_graphs.append((modified_graph, to_remove))

    return acyclic_graphs
