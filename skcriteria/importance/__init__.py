#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Sensitivity-based importance checkers.

Importance, in this context, is the effect that an element of a decision
problem has on its outcome: how much the resulting ranking changes when
that element is altered or removed. Depending on the checker, "element"
can mean a criterion, an alternative, or a weight.

The module offers features for automating this kind of sensitivity
analysis and reporting a per-element importance score, without requiring
an ad hoc distance function: every checker reuses
:class:`skcriteria.cmp.RanksComparator`'s own pairwise comparison methods
to compare the perturbed ranking(s) against a reference ranking.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ._base_importance import CriteriaImportanceABC
from .criteria_keep_only_one import CriteriaKeepOnlyOneChecker
from .criteria_leave_one_out import CriteriaLeaveOneOutChecker
from .criteria_oat import CriteriaOATChecker

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "CriteriaImportanceABC",
    "CriteriaLeaveOneOutChecker",
    "CriteriaKeepOnlyOneChecker",
    "CriteriaOATChecker",
]
