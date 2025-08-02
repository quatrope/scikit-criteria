#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The module 'skcriteria.agg.similarity' is deprecated since v0.9 and will \
be removed in v1.0. Please use 'skcriteria.agg.topsis' instead."""

# =============================================================================
# IMPORTS
# =============================================================================

from .topsis import TOPSIS, topsis
from ..utils import deprecate


# =============================================================================
# WARNINGS
# =============================================================================

deprecate.warn(
    "The module 'skcriteria.agg.similarity' is deprecated since v0.9 and "
    "will be removed in v1.0. Please use 'skcriteria.agg.topsis' instead."
)

# =============================================================================
# ALL
# =============================================================================

__all__ = ["TOPSIS", "topsis"]
