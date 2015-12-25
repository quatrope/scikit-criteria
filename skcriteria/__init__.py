#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOCS
# =============================================================================

"""Scikit-Criteria is a collections of algorithms, methods and techniques for
multiple-criteria decision analysis [].

"""

# =============================================================================
# CONSTANTS
# =============================================================================

__version__ = ("0", "0", "2")

NAME = "scikit-criteria"

DOC = __doc__

VERSION = ".".join(__version__)

AUTHORS = "Cabral & Luczywo"

EMAIL = "jbc.develop@gmail.com"

URL = "http://scikit-criteria.org/"

LICENSE = "3 Clause BSD"

KEYWORDS = "mcda mcdm ahp moora muti criteria".split()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
