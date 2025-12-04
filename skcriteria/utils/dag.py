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
DAG conversion and topological sorting utilities.

This module provides utilities for converting directed graphs to Directed
Acyclic Graphs (DAGs) using the Feedback Arc Set (FAS) algorithm and
generating all possible rankings from topological sorts.

Key Features
------------
- Optimal cycle removal using Integer Programming or Eades heuristic
- Complete enumeration of all valid topological orderings
- Automatic method selection based on graph size
"""

# =============================================================================
# IMPORTS
# =============================================================================

import igraph as ig
import networkx as nx
import numpy as np

# =============================================================================
# FUNCTIONS
# =============================================================================


def _resolve_method(graph, method):
    """Resolve the feedback arc set method based on graph size.

    Automatically selects the optimal method for finding feedback arc sets
    based on the graph size. For smaller graphs (< 100 nodes), it uses the
    exact integer programming method, otherwise defaults to the Eades
    heuristic for better performance on larger graphs.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph for which to resolve the method.
    method : str
        The method to use. If "auto", the method is selected based on
        graph size. Otherwise, the method is returned as-is.

    Returns
    -------
    str
        The resolved method name. Returns "ip" for graphs with less than
        100 nodes when method is "auto", "eades" for larger graphs when
        method is "auto", otherwise returns the input method unchanged.

    """
    # Use exact method (IP) for small graphs, heuristic (Eades) for large ones
    if method == "auto":
        method = "ip" if graph.number_of_nodes() < 100 else "eades"
    return method


def as_dag(graph, method="auto"):
    """Convert a directed graph to a Directed Acyclic Graph (DAG).

    Transforms any directed graph into a DAG by identifying and removing
    the minimum set of edges that form cycles (feedback arc set). The
    algorithm guarantees the result is acyclic by breaking all cycles
    in the graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The directed graph to convert to a DAG. Can be cyclic or acyclic.
    method : str, default "auto"
        The method to use for finding the feedback arc set. Options are:
        - "auto": Automatically selects "ip" for graphs with < 100 nodes,
          otherwise uses "eades" for better performance on larger graphs.
        - "ip": Integer Programming - exact method that finds the minimum
          feedback arc set. Slower but optimal.
        - "eades": Heuristic method by Eades et al. Faster but may remove
          more edges than strictly necessary.

    Returns
    -------
    dag : networkx.DiGraph
        The resulting directed acyclic graph with feedback arcs removed.
    fas : list of int
        The indices of edges that were removed to break all cycles.
    method : str or None
        The method that was actually used. None if the graph was already
        a DAG, otherwise the resolved method name.

    Notes
    -----
    The function guarantees that the returned graph is acyclic by removing
    all edges that participate in cycles. The feedback arc set theorem
    ensures that removing these edges is both necessary and sufficient to
    eliminate all cycles.

    """
    # Check if already a DAG - no processing needed
    if nx.is_directed_acyclic_graph(graph):
        return graph, [], None

    # Resolve method based on graph size for optimal performance
    method = _resolve_method(graph, method)

    # Convert to igraph for efficient FAS computation
    igraph = ig.Graph.from_networkx(graph)

    # Find minimum set of edges that form cycles
    fas = igraph.feedback_arc_set(method=method)

    # Remove feedback arcs to break all cycles
    igraph.delete_edges(fas)

    # Convert back to NetworkX format
    dag = igraph.to_networkx()

    return dag, fas, method


def get_all_rankings(alternatives, dag):
    """Get all possible rankings from a DAG's topological sorts.

    Generates all possible rankings by enumerating every topological sort
    of the DAG. Each ranking represents the position (1-indexed) of each
    alternative in a valid topological ordering.

    Parameters
    ----------
    alternatives : np.ndarray
        Array of alternative names in the original order.
    dag : networkx.DiGraph
        The directed acyclic graph representing preference relations.

    Returns
    -------
    tuple of np.ndarray
        Tuple of rankings, where each ranking is a 1-indexed array containing
        the position of each alternative in that topological sort. The order
        of positions corresponds to the order of alternatives in the input.

    """
    ranks = []
    for sort in nx.all_topological_sorts(dag):
        # Create mapping from value to position in this sort (0-indexed)
        position_map = {value: idx for idx, value in enumerate(sort)}

        # Get positions for each alternative and convert to 1-indexed
        order = np.array([position_map[alt] for alt in alternatives]) + 1

        ranks.append(order)

    return tuple(ranks)
