#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Input/Output utilities for scikit-criteria.

This module provides functions for reading and writing DecisionMatrix objects.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .dmsy import read_dmsy, to_dmsy


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "read_dmsy",
    "to_dmsy",
]
