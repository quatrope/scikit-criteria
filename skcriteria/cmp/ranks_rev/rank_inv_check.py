#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Test Criterion #1 for evaluating the effectiveness MCDA method.

According to this criterion, the best alternative identified by the method
should remain unchanged when a non-optimal alternative is replaced by a
worse alternative, provided that the relative importance of each decision
criterion remains the same.



"""

# =============================================================================
# IMPORTS
# =============================================================================

# import the real agg package
from ...ranksrev.rank_invariant_check import RankInvariantChecker  # noqa
