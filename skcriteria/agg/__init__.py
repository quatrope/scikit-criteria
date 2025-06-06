#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""MCDA aggregation methods and internal machinery."""


# =============================================================================
# IMPORTS
# =============================================================================

from ._agg_base import (
    KernelResult,
    RankResult,
    ResultABC,
    SKCDecisionMakerABC,
)
from .mabac import MABAC

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "KernelResult",
    "RankResult",
    "ResultABC",
    "SKCDecisionMakerABC",
    "MABAC"
    ]
