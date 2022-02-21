#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for skcriteria."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import lp, rank
from .accabc import AccessorABC
from .bunch import Bunch
from .decorators import deprecated, doc_inherit

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "AccessorABC",
    "doc_inherit",
    "deprecated",
    "rank",
    "Bunch",
    "lp",
    "dominance",
]
