#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities and structures of skcriteria."""

# =============================================================================
# IMPORTS
# =============================================================================

from .data import (
    DecisionMatrix,
    KernelResult,
    Objective,
    RankResult,
    ResultABC,
    mkdm,
)
from .methods import (
    SKCDecisionMakerABC,
    SKCMatrixAndWeightTransformerABC,
    SKCMethodABC,
    SKCTransformerABC,
    SKCWeighterABC,
)
from .plot import DecisionMatrixPlotter

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "mkdm",
    "DecisionMatrix",
    "DecisionMatrixPlotter",
    "KernelResult",
    "Objective",
    "RankResult",
    "ResultABC",
    "SKCDecisionMakerABC",
    "SKCMatrixAndWeightTransformerABC",
    "SKCMethodABC",
    "SKCTransformerABC",
    "SKCWeighterABC",
]
