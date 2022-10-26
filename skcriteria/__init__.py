#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Scikit-Criteria is a collections of algorithms, methods and \
techniques for multiple-criteria decision analysis."""

# =============================================================================
# IMPORTS
# =============================================================================

import importlib.metadata

from . import datasets
from .core import DecisionMatrix, Objective, mkdm


# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = ["mkdm", "DecisionMatrix", "Objective", "datasets"]


NAME = "scikit-criteria"

DOC = __doc__

VERSION = importlib.metadata.version(NAME)

__version__ = tuple(VERSION.split("."))


del importlib
