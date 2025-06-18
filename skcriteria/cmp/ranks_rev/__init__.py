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

This Deprecated backward compatibility layer around
skcriteria.ranksrev.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .rank_inv_check import RankInvariantChecker
from ...utils import deprecate

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "RankInvariantChecker",
]


# this will be used in two places
deprecation_reason = (
    "'skcriteria.cmp.ranks_rev' package is deprecated, "
    "use 'skcriteria.ranksrev' instead"
)


deprecate.warn(deprecation_reason)

__doc__ = deprecate.add_sphinx_deprecated_directive(
    __doc__, version="0.9", reason=deprecation_reason
)


# delete the unused modules and variables
del deprecation_reason
