#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.dag_rank"""


# =============================================================================
# IMPORTS
# =============================================================================

import networkx as nx

import numpy as np

import pandas as pd

from skcriteria.utils import dag_rank


# =============================================================================
# TESTS
# =============================================================================


def test_as_condensed_reduced_dag_is_dag():
    # Simple linear chain: A -> B -> C, no cycles
    adj_matrix = pd.DataFrame(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    assert nx.is_directed_acyclic_graph(dag)
    assert dag.number_of_nodes() == 3
    assert all(len(m) == 1 for m in members.values())


def test_as_condensed_reduced_dag_with_cycle():
    # A -> B -> C -> A forms a dominance cycle, all tied into one supernode
    adj_matrix = pd.DataFrame(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    assert nx.is_directed_acyclic_graph(dag)
    assert dag.number_of_nodes() == 1

    (node_members,) = members.values()
    assert set(node_members) == {"A", "B", "C"}


def test_ranking_from_generations():
    # Structure: A -> C, A -> E, C -> B, C -> E, E -> D
    # Generations: [A] -> [C] -> [B, E] -> [D]
    adj_matrix = pd.DataFrame(
        [
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        index=["A", "B", "C", "D", "E"],
        columns=["A", "B", "C", "D", "E"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    result = dag_rank.ranking_from_generations(
        ["A", "B", "C", "D", "E"], dag, members
    )

    # A is in generation 1 (best), C is in generation 2,
    # B and E are tied in generation 3, D is in generation 4 (worst)
    np.testing.assert_array_equal(result, [1, 3, 2, 4, 3])


def test_ranking_from_generations_linear():
    # Simple linear chain: A -> B -> C
    adj_matrix = pd.DataFrame(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    result = dag_rank.ranking_from_generations(["A", "B", "C"], dag, members)

    # Each element in its own generation: A=1 (best), B=2, C=3 (worst)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_ranking_from_generations_all_tied():
    # No edges - all alternatives are incomparable (same generation)
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C"])
    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    result = dag_rank.ranking_from_generations(["A", "B", "C"], dag, members)

    # All in same generation, all rank 1
    np.testing.assert_array_equal(result, [1, 1, 1])


def test_ranking_from_generations_cycle_tied():
    # A -> B -> C -> A forms a cycle: all tied at rank 1
    adj_matrix = pd.DataFrame(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    result = dag_rank.ranking_from_generations(["A", "B", "C"], dag, members)

    np.testing.assert_array_equal(result, [1, 1, 1])


def test_generate_rankings_with_cycle_permutations():
    # A -> {B, C} tournament with B, C tied (cycle B<->C), A always best
    adj_matrix = pd.DataFrame(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    result = dag_rank.generate_rankings_with_cycle_permutations(
        ["A", "B", "C"], dag, members
    )

    rankings = {tuple(r) for r in result}
    assert rankings == {(1, 2, 3), (1, 3, 2)}


def test_generate_rankings_with_cycle_permutations_max_ranks():
    adj_matrix = pd.DataFrame(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    dag, members = dag_rank.as_condensed_reduced_dag(graph)

    result = dag_rank.generate_rankings_with_cycle_permutations(
        ["A", "B", "C"], dag, members, max_rankings=1
    )

    rankings = list(result)
    assert len(rankings) == 1
