#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities for skcriteria.

- Base classes.
- Data types.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from ._base import (
    SKCBaseDecisionMaker,
    SKCDataValidatorMixin,
    SKCMatrixAndWeightTransformerMixin,
    SKCRankerMixin,
    SKCTransformerMixin,
    SKCWeighterMixin,
)
from ._data import DecisionMatrix, Objective, mkdm

# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "DecisionMatrix",
    "Objective",
    "SKCBaseDecisionMaker",
    "SKCDataValidatorMixin",
    "SKCMatrixAndWeightTransformerMixin",
    "SKCRankerMixin",
    "SKCTransformerMixin",
    "SKCWeighterMixin",
    "mkdm",
]
