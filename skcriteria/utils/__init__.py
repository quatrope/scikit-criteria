#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
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
from .cmanagers import df_temporal_header, hidden
from .cycle_removal import generate_acyclic_graphs
from .deprecate import deprecated, will_change
from .dict_cmp import dict_allclose
from .doctools import doc_inherit
from .object_diff import DiffEqualityMixin, diff
from .unames import unique_names


# =============================================================================
# ALL
# =============================================================================

__all__ = [
    "AccessorABC",
    "generate_acyclic_graphs",
    "doc_inherit",
    "deprecated",
    "df_temporal_header",
    "hidden",
    "rank",
    "Bunch",
    "lp",
    "unique_names",
    "will_change",
    "diff",
    "DiffEqualityMixin",
    "dict_allclose",
]
