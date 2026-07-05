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

:func:`as_condensed_reduced_dag` collapses each strongly connected component
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

import itertools as it

import networkx as nx

import numpy as np

# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def as_condensed_reduced_dag(graph):
    """Collapse every strongly connected component into a single node.

    Produces a DAG where each node represents either a single
    alternative or, if it was part of a dominance cycle (a strongly
    connected component of size greater than one), the whole group of
    alternatives tied together by that cycle. The condensation is then
    transitively reduced, dropping edges implied by other paths so only
    the minimal set of edges needed to preserve reachability remains.

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
        The transitive reduction of the condensation of ``graph``: one
        node per strongly connected component, with an edge between two
        supernodes only where required to preserve the reachability
        relation implied by the original graph.
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

    # collapse each strongly connected component (dominance cycle)
    # into a single supernode, then drop transitively redundant edges
    condensed = nx.condensation(graph)
    dag = nx.transitive_reduction(condensed)

    # transitive_reduction discards node attributes, so rebuild the
    # members mapping and give each supernode a readable label
    members, labels, collapsed = {}, {}, False
    for node, data in dag.nodes(data=True):

        # alternatives condensed into this supernode
        node_members = data["members"]
        node_name = "+\n".join(node_members)

        # keyed by the new label, since nodes are relabeled below
        members[node_name] = node_members
        labels[node] = node_name

    # swap the integer SCC ids for the readable labels built above
    nx.relabel_nodes(dag, labels, copy=False)

    return dag, members



def ranking_from_generations(alternatives, dag, members):
    """Generate a ranking based on topological generations.

    Creates a single ranking where alternatives in the same topological
    generation (incomparable elements) share the same rank. This provides
    a compact representation when ties are acceptable.

    Meant to be used with the condensed DAG from :func:`as_condensed_reduced_dag`,
    where a node may represent several alternatives tied together by a
    dominance cycle.

    Parameters
    ----------
    alternatives : array-like
        Array of alternative names/identifiers in their original order.
        This defines the order in which ranks are returned in the ranking.
    dag : networkx.DiGraph
        A directed acyclic graph representing preference relations, as
        returned by :func:`as_condensed_reduced_dag`.
    members : dict
        Maps each node of ``dag`` to the set of alternatives it
        represents, as returned by :func:`as_condensed_reduced_dag`.

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


# [CLAUDE cambia el nombre de la funcion]
def generate_rankings_from_x(alternatives, dag, members, *, max_ranks=None):
    """[CLAUDE COMPLETA]
    """
    # [CLAUDE COMPLETA]
    all_permutations = []
    for generation in nx.topological_generations(dag):

        # [CLAUDE COMPLETA]
        gen_members = members[generation[0]]

        # [CLAUDE COMPLETA]
        generation_permutations = it.permutations(gen_members)
        all_permutations.append(generation_permutations)

    # [CLAUDE COMPLETA]
    generated_rankins = 0
    for permutation in it.product(*all_permutations):
        if max_ranks is not None and generated_rankins >= max_ranks:
            break

        # [CLAUDE no me gusta el nombre plain_permutation]
        plain_permutation = it.chain(*permutation)

        alt_to_rank = {alternative: rank for rank, alternative in enumerate(plain_permutation, start=1)}

        ranking = np.array(
            [alt_to_rank[alt] for alt in alternatives],
            dtype=int,
        )

        yield ranking
        generated_rankins += 1