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
DAG conversion and ranking-reconstruction utilities.

This module provides utilities for converting a directed graph (typically
a pairwise dominance graph / tournament) into a Directed Acyclic Graph,
and for reconstructing a ranking from it.

:func:`as_condensed_dag` collapses each strongly connected component
(dominance cycle) into a single supernode via graph condensation, which
is always exact and acyclic -- no heuristic or arbitrary cycle-breaking
choice is involved. :func:`ranking_from_generations` then builds a
ranking from that DAG, where every alternative that was part of the same
dominance cycle ends up tied at the same rank.

Key Features
------------
- Exact, deterministic cycle handling via strongly connected components
- No approximation or arbitrary choices: every dominance cycle in the
  original graph is reported as an explicit tie in the resulting ranking
"""

# =============================================================================
# IMPORTS
# =============================================================================

import networkx as nx

import numpy as np

# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def as_condensed_dag(graph):
    """Collapse every strongly connected component into a single node.

    Produces a DAG where each node represents either a single
    alternative or, if it was part of a dominance cycle (a strongly
    connected component of size greater than one), the whole group of
    alternatives tied together by that cycle.

    Graph condensation is a direct mathematical construction, always
    exact and always acyclic, with no heuristic involved and no
    arbitrary choice to make about which edge to remove to break a
    cycle. Use this when the goal is to *report* where the pairwise
    comparisons fail to determine a strict order, instead of forcing
    one anyway.

    Parameters
    ----------
    graph : networkx.DiGraph
        The (possibly cyclic) graph to condense.

    Returns
    -------
    dag : networkx.DiGraph
        The condensation of ``graph``: one node per strongly connected
        component, with an edge between two supernodes if there was an
        edge between any of their members in the original graph.
    members : dict
        Maps each node of ``dag`` to the set of original alternatives it
        represents. Nodes coming from a size-one component map to a
        singleton set; nodes coming from a dominance cycle map to all
        the alternatives tied together by that cycle. Meant to be passed
        as the ``members`` argument of :func:`ranking_from_generations`.

    Notes
    -----
    ``nx.transitive_reduction`` does not preserve node attributes, so
    the ``members`` mapping has to be rebuilt after calling it -- this
    is handled internally, callers do not need to worry about it.

    """
    condensed = nx.condensation(graph)
    dag = nx.transitive_reduction(condensed)
    dag.add_nodes_from(condensed.nodes(data=True))
    members = {
        node: data["members"] for node, data in dag.nodes(data=True)
    }
    return dag, members


def generate_rankings_from_toposorts(
    alternatives, dag, members, *, max_rankings=None
):
    """Generate all possible rankings from a DAG's topological sorts.

    Enumerates all valid rankings by computing every possible topological
    sort of the DAG. Each ranking represents a complete ordering of
    alternatives that respects the preference relations encoded in the DAG.

    Meant to be used with the condensed DAG from :func:`as_condensed_dag`,
    where a node may represent several alternatives tied together by a
    dominance cycle: all alternatives belonging to the same supernode
    always receive the same rank in every generated ranking, since there
    is no data to order them relative to each other.

    Parameters
    ----------
    alternatives : array-like
        Array of alternative names/identifiers in their original order.
        This defines the order in which ranks are returned in each ranking.
    dag : networkx.DiGraph
        A directed acyclic graph representing preference relations, as
        returned by :func:`as_condensed_dag`.
    members : dict
        Maps each node of ``dag`` to the set of alternatives it
        represents, as returned by :func:`as_condensed_dag`.
    max_rankings : int, optional
        Maximum number of rankings to generate. If None (default), all
        possible rankings are generated. Use this parameter to limit
        computation when the number of topological sorts is very large.

    Yields
    ------
    np.ndarray
        A 1-indexed NumPy array where the i-th element is the rank
        (position) of the i-th alternative. Lower ranks indicate better
        alternatives. Alternatives tied together in the same supernode
        (dominance cycle) always share the same rank across every
        yielded ranking.

    Notes
    -----
    - The number of rankings can grow exponentially with the number of
      supernodes, especially for DAGs with many incomparable elements.
    - Rankings are 1-indexed (best alternative has rank 1, not 0).
    - This enumerates orderings of the DAG's nodes (supernodes) -- it
      never invents an order *within* a supernode, since a dominance
      cycle means the pairwise comparisons genuinely do not determine
      one.

    """
    rankings_generated = 0

    for topological_order in nx.all_topological_sorts(dag):
        if max_rankings is not None and rankings_generated >= max_rankings:
            break

        # Map each node to its 1-indexed position in this topological sort
        node_to_rank = {
            node: rank for rank, node in enumerate(topological_order, start=1)
        }

        # Expand each node into every alternative it represents, all
        # sharing that node's rank
        alt_to_rank = {
            alt: node_to_rank[node]
            for node in topological_order
            for alt in members[node]
        }

        # Build rank array for all alternatives in original order
        ranking = np.array(
            [alt_to_rank[alt] for alt in alternatives],
            dtype=int,
        )

        yield ranking
        rankings_generated += 1


def ranking_from_generations(alternatives, dag, members):
    """Generate a ranking based on topological generations.

    Creates a single ranking where alternatives in the same topological
    generation (incomparable elements) share the same rank. This provides
    a compact representation when ties are acceptable.

    Meant to be used with the condensed DAG from :func:`as_condensed_dag`,
    where a node may represent several alternatives tied together by a
    dominance cycle.

    Parameters
    ----------
    alternatives : array-like
        Array of alternative names/identifiers in their original order.
        This defines the order in which ranks are returned in the ranking.
    dag : networkx.DiGraph
        A directed acyclic graph representing preference relations, as
        returned by :func:`as_condensed_dag`.
    members : dict
        Maps each node of ``dag`` to the set of alternatives it
        represents, as returned by :func:`as_condensed_dag`.

    Returns
    -------
    np.ndarray
        A 1-indexed NumPy array where the i-th element is the rank of the
        i-th alternative. Alternatives in the same generation (i.e. in
        the same dominance cycle) share the same rank. Lower ranks
        indicate better alternatives.

    """
    # Map each alternative to its generation number (1-indexed)
    alt_to_rank = {}
    for rank, generation in enumerate(
        nx.topological_generations(dag), start=1
    ):
        for c_node in generation:
            for alt in members[c_node]:
                alt_to_rank[alt] = rank

    # Build rank array in original alternative order
    ranking = np.array(
        [alt_to_rank[alt] for alt in alternatives],
        dtype=int,
    )
    return ranking