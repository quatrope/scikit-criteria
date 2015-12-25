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

"""Utilities for scikit-criteria

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

MIN = -1

MAX = 1


# =============================================================================
# FUNCTIONS
# =============================================================================

def criteriarr(criteria):
    criteria = np.asarray(criteria)
    if np.setdiff1d(criteria, [MIN, MAX]):
        msg = "Criteria Array only accept '{}' or '{}' Values. Found {}"
        raise ValueError(msg.format(MAX, MIN, criteria))
    return criteria


def is_mtx(mtx, size=None):
    try:
        a, b = mtx.shape
        if size and (a, b) != size:
            return False
    except:
        return False
    return True
