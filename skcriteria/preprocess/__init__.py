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

from .invert_objectives import MinimizeToMaximize, invert
from .max_scaler import MaxScaler, scale_by_max
from .minmax_scaler import MinMaxScaler, scale_by_minmax
from .sum_scaler import SumScaler, scale_by_sum
from .vector_scaler import VectorScaler, scale_by_vector

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "MinimizeToMaximize",
    "invert",
    "MaxScaler",
    "scale_by_max",
    "MinMaxScaler",
    "scale_by_minmax",
    "SumScaler",
    "scale_by_sum",
    "VectorScaler",
    "scale_by_vector",
]
