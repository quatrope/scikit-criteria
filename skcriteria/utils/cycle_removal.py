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

import random
import warnings
from collections import Counter
from itertools import combinations

import networkx as nx

# =============================================================================
# FUNCTIONS
# =============================================================================


def _cycle_to_edges(cycle):
    """Convert a cycle (list of nodes) to a list of edges."""
    return [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]


def _select_edge_random(cycle, edge_freq, rng):
    """Select a random edge from cycle."""
    edges = _cycle_to_edges(cycle)
    return rng.choice(edges)


def _select_edge_weighted(cycle, edge_freq, rng):
    """Select edge with probability proportional to frequency (more frequent = more likely)."""
    edges = _cycle_to_edges(cycle)
    weights = [edge_freq[edge] + 1 for edge in edges]
    return rng.choices(edges, weights=weights, k=1)[0]


def filter_minimal_removals(acyclic_graphs):
    """
    Filter acyclic graphs to keep only those with minimal edge removals.
    
    Parameters
    ----------
    acyclic_graphs : list
        List of tuples (acyclic_graph, removed_edges_set)
        
    Returns
    -------
    list
        Filtered list with only minimal removal solutions
    """
    if not acyclic_graphs:
        return []
    
    to_discard = set()

    for (i1, (g1, r1)), (i2, (g2, r2)) in combinations(enumerate(acyclic_graphs), 2):
        if r1.issubset(r2) and r1 != r2:
            to_discard.add(i2)
        elif r2.issubset(r1) and r1 != r2:
            to_discard.add(i1)

    return [gr for i, gr in enumerate(acyclic_graphs) if i not in to_discard]


def generate_acyclic_graphs(graph, strategy="random", max_attempts=1000, max_graphs=10, seed=42):
    """
    Generate multiple acyclic graphs by removing edges from cycles.
    
    Parameters
    ----------
    graph : networkx.DiGraph
        The input directed graph
    strategy : str, optional
        Edge selection strategy: "random" or "weighted"
    max_attempts : int, optional
        Maximum number of attempts to generate graphs
    max_graphs : int, optional
        Maximum number of acyclic graphs to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list
        List of tuples (acyclic_graph, removed_edges_set)
    """
    rng = random.Random(seed)
    seen_removals = set()
    acyclic_graphs = []

    # Select strategy function and precalculate frequencies if needed
    if strategy == "random":
        select_edge = _select_edge_random
        edge_freq = Counter()  # Empty counter for consistency
    elif strategy == "weighted":
        select_edge = _select_edge_weighted
        # Precalculate edge frequencies for weighted strategy
        edge_freq = Counter()
        for cycle in nx.simple_cycles(graph):
            edges = _cycle_to_edges(cycle)
            edge_freq.update(edges)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'random' or 'weighted'.")

    attempts = 0
    while attempts < max_attempts and len(acyclic_graphs) < max_graphs:
        attempts += 1
        to_remove = set()

        cycles = list(nx.simple_cycles(graph))
        if not cycles: # TODO: Corregir
            break

        # Select edges to remove from each cycle
        for cycle in cycles:
            edge_to_remove = select_edge(cycle, edge_freq, rng)
            to_remove.add(edge_to_remove)
        
        # Skip if we've seen this combination before
        removal_frozenset = frozenset(to_remove)
        if removal_frozenset in seen_removals:
            continue
        
        seen_removals.add(removal_frozenset)
        
        # Test if removing edges creates acyclic graph
        modified_graph = graph.copy()
        modified_graph.remove_edges_from(to_remove)
        
        if nx.is_directed_acyclic_graph(modified_graph):
            acyclic_graphs.append((modified_graph, to_remove))
            # Filter to keep only minimal removals
            acyclic_graphs = filter_minimal_removals(acyclic_graphs)

    return acyclic_graphs