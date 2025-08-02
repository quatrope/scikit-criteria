#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The 'skcriteria.pipeline' module is deprecated since 0. and will be \
removed in 1.0 Use 'skcriteria.pipelines' instead."""

# =============================================================================
# IMPORTS
# =============================================================================

from .pipelines import SKCPipeline, mkpipe
from .utils import deprecate


# =============================================================================
# WARNINGS
# =============================================================================

deprecate.warn(
    "The 'skcriteria.pipeline' module is deprecated since 0.9 and will be "
    "removed in 1.0 Use 'skcriteria.pipelines' instead.",
)

# =============================================================================
# ALL
# =============================================================================

__all__ = ["SKCPipeline", "mkpipe"]
