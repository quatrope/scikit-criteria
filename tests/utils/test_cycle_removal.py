#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for cycle_removal"""

# =============================================================================
# IMPORTS
# =============================================================================

import networkx as nx

import pytest

from skcriteria.utils import cycle_removal

# =============================================================================
# TESTS
# =============================================================================

# MAIN FUNCTIONALITY TESTS ===================================================


def test_generate_acyclic_graphs_simple_cycle():
    """Test generate_acyclic_graphs with a simple triangular cycle."""

    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=5, seed=42
    )

    assert len(acyclic_graphs) <= 5

    for acyclic_graph, removed_edges in acyclic_graphs:
        # Check that result is acyclic
        assert nx.is_directed_acyclic_graph(acyclic_graph)

        # Check that exactly one edge was removed from the cycle
        assert len(removed_edges) == 1

        # Check that removed edge was from the original cycle
        cycle_edges = {(1, 2), (2, 3), (3, 1)}
        assert removed_edges.issubset(cycle_edges)

        # Check that acyclic graph has correct number of edges
        assert (
            acyclic_graph.number_of_edges()
            == graph.number_of_edges() - len(removed_edges)
        )


def test_generate_acyclic_graphs_multiple_cycles():
    """Test generate_acyclic_graphs with multiple cycles."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1), (2, 4), (4, 2)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=3, seed=42
    )

    assert len(acyclic_graphs) <= 3

    for acyclic_graph, removed_edges in acyclic_graphs:
        # Check that result is acyclic
        assert nx.is_directed_acyclic_graph(acyclic_graph)

        # Should remove at least one edge from each cycle
        assert len(removed_edges) >= 2


def test_generate_acyclic_graphs_already_acyclic():
    """Test generate_acyclic_graphs with an already acyclic graph."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 4)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=5, seed=42
    )

    # Should return one graph with no edges removed
    assert len(acyclic_graphs) == 1
    acyclic_graph, removed_edges = acyclic_graphs[0]

    assert nx.is_directed_acyclic_graph(acyclic_graph)
    assert len(removed_edges) == 0
    assert acyclic_graph.number_of_edges() == graph.number_of_edges()


def test_generate_acyclic_graphs_empty_graph():
    """Test generate_acyclic_graphs with an empty graph."""
    graph = nx.DiGraph([])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=5, seed=42
    )

    assert len(acyclic_graphs) == 1
    acyclic_graph, removed_edges = acyclic_graphs[0]

    assert nx.is_directed_acyclic_graph(acyclic_graph)
    assert len(removed_edges) == 0
    assert acyclic_graph.number_of_nodes() == 0


@pytest.mark.parametrize("strategy", ["random", "weighted"])
def test_generate_acyclic_graphs_strategies(strategy):
    """Test generate_acyclic_graphs with different strategies."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, strategy=strategy, max_graphs=3, seed=42
    )

    assert len(acyclic_graphs) <= 3

    for acyclic_graph, removed_edges in acyclic_graphs:
        assert nx.is_directed_acyclic_graph(acyclic_graph)


def test_generate_acyclic_graphs_invalid_strategy():
    """Test generate_acyclic_graphs with invalid strategy."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])

    with pytest.raises(ValueError, match="Unknown strategy: invalid"):
        cycle_removal.generate_acyclic_graphs(graph, strategy="invalid")


def test_generate_acyclic_graphs_reproducibility():
    """Test that generate_acyclic_graphs produces reproducible results with \
    same seed."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])

    result1 = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=5, seed=42
    )
    result2 = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=5, seed=42
    )

    assert len(result1) == len(result2)

    for (graph1, edges1), (graph2, edges2) in zip(result1, result2):
        assert edges1 == edges2
        assert set(graph1.edges()) == set(graph2.edges())


def test_generate_acyclic_graphs_max_graphs_limit():
    """Test that generate_acyclic_graphs respects max_graphs limit."""
    graph = nx.DiGraph([(1, 2), (2, 3), (3, 1)])

    max_graphs = 2
    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=max_graphs, seed=42
    )

    assert len(acyclic_graphs) <= max_graphs


def test_generate_acyclic_graphs_complex_graph():
    """Test generate_acyclic_graphs with a more complex graph."""
    # Create a graph with multiple interconnected cycles
    edges = [
        (1, 2),
        (2, 3),
        (3, 1),  # First cycle
        (3, 4),
        (4, 5),
        (5, 3),  # Second cycle
        (2, 6),
        (6, 7),
        (7, 2),  # Third cycle
        (1, 8),
        (8, 9),  # Acyclic part
    ]
    graph = nx.DiGraph(edges)

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=5, seed=42
    )

    for acyclic_graph, removed_edges in acyclic_graphs:
        # Check that result is acyclic
        assert nx.is_directed_acyclic_graph(acyclic_graph)

        # Should remove edges from multiple cycles
        assert len(removed_edges) >= 3  # At least one edge per cycle

        # Check that acyclic part is preserved
        assert acyclic_graph.has_edge(1, 8)
        assert acyclic_graph.has_edge(8, 9)


def test_generate_acyclic_graphs_self_loop():
    """Test generate_acyclic_graphs with self-loops."""
    graph = nx.DiGraph([(1, 1), (1, 2), (2, 3)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        graph, max_graphs=3, seed=42
    )

    for acyclic_graph, removed_edges in acyclic_graphs:
        assert nx.is_directed_acyclic_graph(acyclic_graph)
        # Self-loop should be removed
        assert (1, 1) in removed_edges
