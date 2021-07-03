#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Normalization routines."""

# =============================================================================
# IMPORTS
# =============================================================================

from .invert_objective import MinimizeToMaximizeNormalizer, invert
from .max_normalizer import MaxNormalizer, max_norm
from .sum_normalizer import SumNormalizer, sum_norm
from .vector_normalizer import VectorNormalizer, vector_norm

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "MinimizeToMaximizeNormalizer",
    "invert",
    "MaxNormalizer",
    "max_norm",
    "SumNormalizer",
    "sum_norm",
    "VectorNormalizer",
    "vector_norm",
]
