#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for a-posteriori analysis of experiments."""

# =============================================================================
# IMPORTS
# =============================================================================

from .ranks_cmp import RanksComparator, mkrank_cmp
from .ranks_rev import RankInvariantChecker

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "RanksComparator",
    "RankInvariantChecker",
    "mkrank_cmp",
]
