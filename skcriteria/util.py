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
        mtx = np.asarray(mtx)
        a, b = mtx.shape
        if size and (a, b) != size:
            return False
    except:
        return False
    return True


def nearest(array, value, side=None):
    # based on: http://stackoverflow.com/a/2566508
    #           http://stackoverflow.com/a/3230123
    #           http://stackoverflow.com/a/17119267
    if side not in (None, "gt", "lt"):
        msg = "'side' must be None, 'gt' or 'lt'. Found {}".format(side)
        raise ValueError(msg)

    raveled = np.ravel(array)
    cleaned = raveled[~np.isnan(raveled)]

    if side is None:
        idx = np.argmin(np.abs(cleaned-value))

    else:
        masker, decisor = (
            (np.ma.less_equal,  np.argmin)
            if side == "gt" else
            (np.ma.greater_equal, np.argmax))

        diff = cleaned - value
        mask = masker(diff, 0)
        if np.all(mask):
            return None

        masked_diff = np.ma.masked_array(diff, mask)
        idx = decisor(masked_diff)

    return cleaned[idx]
