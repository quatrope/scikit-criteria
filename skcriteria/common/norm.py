#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> and <nluczywo@gmail.com>
# wrote this file. As long as you retain this notice you can do whatever you
# want with this stuff. If we meet some day, and you think this stuff is worth
# it, you can buy me a WISKEY in return Juan BC and Nadia AL.


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


#==============================================================================
# DOCS
#==============================================================================

__doc__  = """Several implementations of normalization methods

"""


#==============================================================================
# IMPORTS
#==============================================================================

import numpy as np


#==============================================================================
# IMPLEMENTATIONS
#==============================================================================

def sum(arr, axis=None):
    colsum = np.sum(arr, axis=axis)
    return np.divide(arr, colsum, dtype="f")


def max(arr, axis=None):
    colmax = np.max(arr, axis=axis)
    return np.divide(arr, colmax, dtype="f")


def vector(arr, axis=None):
    sqrt = np.sqrt(np.power(arr, 2).sum(axis=axis))
    return np.divide(arr, sqrt, dtype="f")


def push_negatives(arr, axis=None):
    mins = np.min(arr, axis=axis)
    delta = (mins < 0) * mins
    return np.subtract(arr, delta)


def eps(arr, axis=None):
    eps = 0
    arr = np.asarray(arr)
    if np.any(arr == 0):
        if issubclass(arr.dtype.type, (np.inexact, float)):
            eps = np.finfo(arr.dtype.type).eps
        else:
            eps = np.finfo(np.float).eps
    return arr + eps


#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    print(__doc__)
