#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities and structures of skcriteria."""

# =============================================================================
# IMPORTS
# =============================================================================

from .data import DecisionMatrix, mkdm
from .methods import SKCMethodABC
from .objectives import Objective
from .plot import DecisionMatrixPlotter

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "mkdm",
    "DecisionMatrix",
    "DecisionMatrixPlotter",
    "Objective",
    "SKCMethodABC",
]
