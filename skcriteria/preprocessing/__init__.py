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

from .add_value_to_zero import AddValueToZero, add_value_to_zero
from .invert_objectives import MinimizeToMaximize, invert
from .push_negatives import PushNegatives, push_negatives
from .scalers import (
    MaxScaler,
    MinMaxScaler,
    StandarScaler,
    SumScaler,
    VectorScaler,
    scale_by_max,
    scale_by_minmax,
    scale_by_stdscore,
    scale_by_sum,
    scale_by_vector,
)

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "AddValueToZero",
    "add_value_to_zero",
    "MinimizeToMaximize",
    "invert",
    "MaxScaler",
    "scale_by_max",
    "MinMaxScaler",
    "scale_by_minmax",
    "PushNegatives",
    "push_negatives",
    "StandarScaler",
    "scale_by_stdscore",
    "SumScaler",
    "scale_by_sum",
    "VectorScaler",
    "scale_by_vector",
]
