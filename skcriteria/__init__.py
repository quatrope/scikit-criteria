#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
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
from .io import read_dmsy, to_dmsy
from .utils.ondemand_import import mk_ondemand_importer_for


# =============================================================================
# CONSTANTS
# =============================================================================

__all__ = [
    "mkdm",
    "DecisionMatrix",
    "Objective",
    "datasets",
    "read_dmsy",
    "to_dmsy",
    "CombinatorialPipeline",
]


NAME = "scikit-criteria"

DOC = __doc__

VERSION = importlib.metadata.version(NAME)

__version__ = tuple(VERSION.split("."))


# Patch __getattr__ and __dir__ to use ondemand_importer
# This enable the laxy loading of submodules without importing them
ondemand_importer = mk_ondemand_importer_for("skcriteria")
__getattr__ = ondemand_importer.import_or_get_attribute
__dir__ = ondemand_importer.list_available_modules


# delete the unused modules and variables
del importlib.metadata, ondemand_importer
