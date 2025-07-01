#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Rank reversal tools.

Rank reversal is a change in the preferred order of alternatives that occurs
when the selection method or available options change. It is a significant
issue in decision-making, particularly in multi-criteria decision-making.

One way to test the validity of decision-making methods is to construct special
test problems and then study the solutions they derive. If the solutions
exhibit some logic contradictions (in the form of undesirable rank reversals
of the alternatives), then one may argue that something is wrong with the
method that derived them.

The module offers features for automating the execution and assessment of
standard tests for rank reversal, primarily focusing on alterations in the
available options.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .rank_invariant_check import RankInvariantChecker
from .rank_transitivity_check import RankTransitivityChecker

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "RankInvariantChecker",
    "RankTransitivityChecker",
]
