#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for skcriteria."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import lp, rank
from .bunch import Bunch
from .decorators import doc_inherit

# =============================================================================
# ALL
# =============================================================================

__all__ = ["doc_inherit", "rank", "Bunch", "lp", "dominance"]
