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
# PRIVATE HELPERS
# =============================================================================


def _nx_fas(igraph, fas):
    """Convert igraph edge IDs to NetworkX edge tuples.

    Translates a feedback arc set represented as igraph edge IDs into
    NetworkX-compatible edge tuples using the original node names.

    Parameters
    ----------
    igraph : igraph.Graph
        The igraph representation of the graph, where nodes have
        "_nx_name" attributes storing their original NetworkX names.
    fas : list of int
        List of igraph edge IDs that form the feedback arc set.

    Returns
    -------
    list of tuple
        List of NetworkX edge tuples (source, target) corresponding to
        the feedback arc set edges.

    Notes
    -----
    This function is used internally to map between igraph's edge
    representation (integer IDs) and NetworkX's edge representation
    (node name tuples), preserving the original node names from the
    NetworkX graph.

    """
    nx_edges = []
    for edge in igraph.es[fas]:
        nx_edge = tuple(
            node.attributes()["_nx_name"] for node in edge.vertex_tuple
        )
        nx_edges.append(nx_edge)
    return nx_edges


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def resolve_fas_method(graph, method):
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


def as_dag(graph, *, method="auto") -> list[nx.DiGraph, list, str | None]:
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
    method = resolve_fas_method(graph, method)

    # Convert to igraph for efficient FAS computation
    igraph = ig.Graph.from_networkx(graph)

    # Find minimum set of edges that form cycles
    fas = igraph.feedback_arc_set(method=method)

    # get the nodes between edges before removal
    nx_fas = _nx_fas(igraph, fas)

    # Remove feedback arcs to break all cycles
    igraph.delete_edges(fas)

    # Convert back to NetworkX format
    dag = igraph.to_networkx()

    return dag, nx_fas, method


def all_rankings(alternatives, dag, *, max_rankings=None):
    """Generate all possible rankings from a DAG's topological sorts.

    Enumerates all valid rankings by computing every possible topological
    sort of the DAG. Each ranking represents a complete ordering of
    alternatives that respects the preference relations encoded in the DAG.

    Parameters
    ----------
    alternatives : array-like
        Array of alternative names/identifiers in their original order.
        This defines the order in which ranks are returned in each ranking.
    dag : networkx.DiGraph
        A directed acyclic graph representing preference relations between
        alternatives. Edges point from preferred to less preferred
        alternatives.
    max_rankings : int, optional
        Maximum number of rankings to generate. If None (default), all
        possible rankings are generated. Use this parameter to limit
        computation when the number of topological sorts is very large.

    Yields
    ------
    np.ndarray
        A 1-indexed NumPy array where the i-th element is the rank (position)
        of the i-th alternative. Lower ranks indicate better alternatives.
        Alternatives not present in the DAG are assigned a rank of
        len(alternatives) + 1.

    Notes
    -----
    - The number of rankings can grow exponentially with the number of
      alternatives, especially for DAGs with many incomparable elements.
    - Rankings are 1-indexed (best alternative has rank 1, not 0).
    - Alternatives missing from the DAG (e.g., filtered out) receive the
      last possible rank position (len(alternatives) + 1).
    - When max_rankings is specified, the function stops generating rankings
      once the limit is reached, which can significantly reduce computation
      time for large DAGs.

    """
    rankings_generated = 0

    for topological_order in nx.all_topological_sorts(dag):
        if max_rankings is not None and rankings_generated >= max_rankings:
            break

        # Map each alternative to its 1-indexed position in this topological sort
        alternative_to_rank = {
            alt: rank for rank, alt in enumerate(topological_order, start=1)
        }

        # Build rank array for all alternatives in original order

        ranking = np.array(
            [alternative_to_rank[alt] for alt in alternatives],
            dtype=int,
        )

        yield ranking
        rankings_generated += 1
