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

import pytest

from skcriteria.utils import dag_rank


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    "nodes, method, expected",
    [
        (10, "auto", "ip"),
        (99, "auto", "ip"),
        (99, "eades", "eades"),
        (100, "auto", "eades"),
        (100, "ip", "ip"),
    ],
)
def test_resolve_fas_method(nodes, method, expected):
    graph = nx.erdos_renyi_graph(n=nodes, p=0.3)
    result = dag_rank.resolve_fas_method(graph, method)
    assert result == expected


def test_as_dag_not_dag():

    # Crear la matriz de adyacencia como DataFrame
    # Basándome en el diagrama que subiste
    adj_matrix = pd.DataFrame(
        {
            "A": [0, 0, 1, 0, 1],
            "B": [0, 0, 0, 0, 0],
            "C": [1, 1, 0, 0, 1],
            "D": [0, 0, 0, 0, 0],
            "E": [0, 0, 1, 1, 0],
        },
        index=["A", "B", "C", "D", "E"],
    )

    # Crear el grafo dirigido desde el DataFrame
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    r_graph, r_fas, r_method = dag_rank.as_dag(graph, method="auto")

    assert nx.is_directed_acyclic_graph(r_graph)
    assert r_fas == [("A", "C"), ("C", "E")]
    assert r_method == "ip"


def test_as_dag_is_dag():

    # Crear la matriz de adyacencia como DataFrame
    # Basándome en el diagrama que subiste
    adj_matrix = pd.DataFrame(
        {
            "A": [0, 1],
            "B": [0, 0],
        },
        index=["A", "B"],
    )

    # Crear el grafo dirigido desde el DataFrame
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    r_graph, r_fas, r_method = dag_rank.as_dag(graph, method="auto")

    assert nx.is_directed_acyclic_graph(r_graph)
    assert r_fas == []
    assert r_method is None


def test_generate_rankings_from_toposorts():
    adj_matrix = pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 0],
            "B": [0, 0, 1, 0, 0],
            "C": [1, 0, 0, 0, 0],
            "D": [0, 0, 0, 0, 1],
            "E": [1, 0, 1, 0, 0],
        },
        index=["A", "B", "C", "D", "E"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    result = dag_rank.generate_rankings_from_toposorts(["A", "B", "C", "D", "E"], graph)

    np.testing.assert_array_equal(next(result), [1, 5, 2, 4, 3])
    np.testing.assert_array_equal(next(result), [1, 4, 2, 5, 3])
    np.testing.assert_array_equal(next(result), [1, 3, 2, 5, 4])

    with pytest.raises(StopIteration):
        next(result)


def test_generate_rankings_from_toposorts_max1():
    adj_matrix = pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 0],
            "B": [0, 0, 1, 0, 0],
            "C": [1, 0, 0, 0, 0],
            "D": [0, 0, 0, 0, 1],
            "E": [1, 0, 1, 0, 0],
        },
        index=["A", "B", "C", "D", "E"],
    )
    graph = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

    result = dag_rank.generate_rankings_from_toposorts(
        ["A", "B", "C", "D", "E"], graph, max_rankings=1
    )

    np.testing.assert_array_equal(next(result), [1, 5, 2, 4, 3])

    with pytest.raises(StopIteration):
        next(result)
