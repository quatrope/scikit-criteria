#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Scikit-Criteria is a collections of algorithms, methods and
techniques for multiple-criteria decision analysis.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

if os.getenv("__SKCRITERIA_IN_SETUP__") != "True":
    from .data import *  # noqa

del os


# =============================================================================
# CONSTANTS
# =============================================================================

__version__ = ("0", "3")

NAME = "scikit-criteria"

DOC = __doc__

VERSION = ".".join(__version__)

AUTHORS = "Cabral & Luczywo"

EMAIL = "jbc.develop@gmail.com"

URL = "http://scikit-criteria.org/"

LICENSE = "3 Clause BSD"

KEYWORDS = (
    "muticriteria mcda mcdm".split() +
    "weightedsum weightedproduct simus topsis moora electre".split())
