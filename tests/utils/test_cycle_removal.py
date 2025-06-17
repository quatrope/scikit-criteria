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
import numpy as np
import pytest

from skcriteria.utils import cycle_removal

# =============================================================================
# TESTS
# =============================================================================

# UTILITY FUNCTIONS TESTS ====================================================


def test_cycle_to_edges_simple():
    """Test _cycle_to_edges with a simple cycle."""
    cycle = [1, 2, 3]
    expected = [(1, 2), (2, 3), (3, 1)]
    result = cycle_removal._cycle_to_edges(cycle)
    assert result == expected


def test_cycle_to_edges_two_nodes():
    """Test _cycle_to_edges with a two-node cycle."""
    cycle = [1, 2]
    expected = [(1, 2), (2, 1)]
    result = cycle_removal._cycle_to_edges(cycle)
    assert result == expected


def test_calculate_edge_frequencies():
    """Test _calculate_edge_frequencies with a graph containing cycles."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (2, 4), (4, 2)])

    edge_freq = cycle_removal._calculate_edge_frequencies(G)

    # Should have cycles: [1, 2, 3] and [2, 4]
    assert edge_freq[(1, 2)] == 1  # Only in first cycle
    assert edge_freq[(2, 3)] == 1  # Only in first cycle
    assert edge_freq[(3, 1)] == 1  # Only in first cycle
    assert edge_freq[(2, 4)] == 1  # Only in second cycle
    assert edge_freq[(4, 2)] == 1  # Only in second cycle


def test_calculate_edge_frequencies_no_cycles():
    """Test _calculate_edge_frequencies with an acyclic graph."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])

    edge_freq = cycle_removal._calculate_edge_frequencies(G)

    assert len(edge_freq) == 0


def test_select_edge_random():
    """Test _select_edge_random returns a valid edge from the cycle."""
    cycle = [1, 2, 3]
    edge_freq = {}  # Not used in random selection
    rng = np.random.default_rng(42)

    edge = cycle_removal._select_edge_random(cycle, edge_freq, rng)
    expected_edges = [(1, 2), (2, 3), (3, 1)]

    assert edge in expected_edges


def test_select_edge_weighted():
    """Test _select_edge_weighted considers edge frequencies."""
    cycle = [1, 2, 3]
    edge_freq = {(1, 2): 10, (2, 3): 1, (3, 1): 1}  # (1, 2) much more frequent
    rng = np.random.default_rng(42)

    # Run multiple times to check that high-frequency edges are more likely
    selected_edges = []
    for _ in range(100):
        edge = cycle_removal._select_edge_weighted(cycle, edge_freq, rng)
        selected_edges.append(edge)

    # (1, 2) should be selected more often due to higher frequency
    edge_12_count = selected_edges.count((1, 2))
    assert edge_12_count > 50  # Should be selected more than half the time


# MAIN FUNCTIONALITY TESTS ===================================================


def test_generate_acyclic_graphs_simple_cycle():
    """Test generate_acyclic_graphs with a simple triangular cycle."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=5, seed=42
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
        assert acyclic_graph.number_of_edges() == G.number_of_edges() - len(
            removed_edges
        )


def test_generate_acyclic_graphs_multiple_cycles():
    """Test generate_acyclic_graphs with multiple cycles."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (2, 4), (4, 2)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=3, seed=42
    )

    assert len(acyclic_graphs) <= 3

    for acyclic_graph, removed_edges in acyclic_graphs:
        # Check that result is acyclic
        assert nx.is_directed_acyclic_graph(acyclic_graph)

        # Should remove at least one edge from each cycle
        assert len(removed_edges) >= 2


def test_generate_acyclic_graphs_already_acyclic():
    """Test generate_acyclic_graphs with an already acyclic graph."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=5, seed=42
    )

    # Should return one graph with no edges removed
    assert len(acyclic_graphs) == 1
    acyclic_graph, removed_edges = acyclic_graphs[0]

    assert nx.is_directed_acyclic_graph(acyclic_graph)
    assert len(removed_edges) == 0
    assert acyclic_graph.number_of_edges() == G.number_of_edges()


def test_generate_acyclic_graphs_empty_graph():
    """Test generate_acyclic_graphs with an empty graph."""
    G = nx.DiGraph()

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=5, seed=42
    )

    assert len(acyclic_graphs) == 1
    acyclic_graph, removed_edges = acyclic_graphs[0]

    assert nx.is_directed_acyclic_graph(acyclic_graph)
    assert len(removed_edges) == 0
    assert acyclic_graph.number_of_nodes() == 0


@pytest.mark.parametrize("strategy", ["random", "weighted"])
def test_generate_acyclic_graphs_strategies(strategy):
    """Test generate_acyclic_graphs with different strategies."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, strategy=strategy, max_graphs=3, seed=42
    )

    assert len(acyclic_graphs) <= 3

    for acyclic_graph, removed_edges in acyclic_graphs:
        assert nx.is_directed_acyclic_graph(acyclic_graph)


def test_generate_acyclic_graphs_invalid_strategy():
    """Test generate_acyclic_graphs with invalid strategy."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    with pytest.raises(ValueError, match="Unknown strategy: invalid"):
        cycle_removal.generate_acyclic_graphs(G, strategy="invalid")


def test_generate_acyclic_graphs_reproducibility():
    """Test that generate_acyclic_graphs produces reproducible results with same seed."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    result1 = cycle_removal.generate_acyclic_graphs(G, max_graphs=5, seed=42)
    result2 = cycle_removal.generate_acyclic_graphs(G, max_graphs=5, seed=42)

    assert len(result1) == len(result2)

    for (graph1, edges1), (graph2, edges2) in zip(result1, result2):
        assert edges1 == edges2
        assert set(graph1.edges()) == set(graph2.edges())


def test_generate_acyclic_graphs_max_graphs_limit():
    """Test that generate_acyclic_graphs respects max_graphs limit."""
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    max_graphs = 2
    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=max_graphs, seed=42
    )

    assert len(acyclic_graphs) <= max_graphs


def test_generate_acyclic_graphs_complex_graph():
    """Test generate_acyclic_graphs with a more complex graph."""
    G = nx.DiGraph()
    # Create a graph with multiple interconnected cycles
    G.add_edges_from(
        [
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
    )

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=5, seed=42
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
    G = nx.DiGraph()
    G.add_edges_from([(1, 1), (1, 2), (2, 3)])

    acyclic_graphs = cycle_removal.generate_acyclic_graphs(
        G, max_graphs=3, seed=42
    )

    for acyclic_graph, removed_edges in acyclic_graphs:
        assert nx.is_directed_acyclic_graph(acyclic_graph)
        # Self-loop should be removed
        assert (1, 1) in removed_edges
