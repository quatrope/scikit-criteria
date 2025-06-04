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

import warnings

import networkx as nx

# =============================================================================
# FUNCTIONS
# =============================================================================


def break_cycles_greedy(G):
    """Removes cycles from a directed Networkx graph using a greedy algorithm.
    Creates a copy."""
    G = G.copy()
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            G.remove_edge(*cycle[-1][:2])
        except nx.exception.NetworkXNoCycle:
            warnings.warn("No cycles found on graph")
            break
    return G
