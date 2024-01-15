#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to compare two dictionaries with numpy arrays."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# CLASSES
# =============================================================================


def npdict_all_equals(left, right, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return True if the two dictionaries are equal within a tolerance."""

    if left is right:  # if they are the same object return True
        return True

    # extra keys
    keys = set(left).union(right)

    # if the keys are not the same on both sides return False
    if not (len(keys) == len(left) == len(right)):
        return False

    is_equal = True  # flag to check if all keys are equal, optimist
    while is_equal and keys:  # loop until all keys are equal
        key = keys.pop()
        left_value, right_value = left[key], right[key]

        if type(left_value) is not type(right_value):
            is_equal = False

        elif isinstance(left_value, np.ndarray):
            is_equal = np.allclose(
                left_value,
                right_value,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )

        else:
            is_equal = left_value == right_value

    return is_equal
