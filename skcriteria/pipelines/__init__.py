#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The Module implements utilities to build a composite decision-maker."""

# =============================================================================
# IMPORTS
# =============================================================================

from skcriteria.pipelines.combinatorial import (
    SKCCombinatorialPipeline,
    mkcombinatorial,
)
from skcriteria.pipelines.simple_pipeline import SKCPipeline, mkpipe

__all__ = [
    "SKCPipeline",
    "mkpipe",
    "SKCCombinatorialPipeline",
    "mkcombinatorial",
]
