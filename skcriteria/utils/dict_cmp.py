#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to compare two dictionaries with numpy arrays."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections.abc import Mapping

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

_INEXACT_TYPES = (float, complex, np.inexact)

# =============================================================================
# CLASSES
# =============================================================================


def dict_allclose(left, right, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Compares two dictionaries. If values of type "numpy.array" are \
    encountered, the function utilizes "numpy.allclose" for comparison.

    Parameters
    ----------
    left : dict
        The left dictionary.
    right : dict
        The right dictionary.
    rtol : float, optional
        The relative tolerance parameter for `np.allclose`.
    atol : float, optional
        The absolute tolerance parameter for `np.allclose`.
    equal_nan : bool, optional
        Whether to consider NaN values as equal.

    Returns
    -------
    bool
        True if the dictionaries are equal, False otherwise.

    Notes
    -----
    This function iteratively compares the values of corresponding keys in the
    input dictionaries `left` and `right`. It handles various data types,
    including NumPy arrays, and uses the `np.allclose` function for numeric
    array comparisons with customizable tolerance levels. The comparison is
    performed iteratively, and the function returns True if all values are
    equal based on the specified criteria. If the dictionaries have different
    lengths or keys, or if the types of corresponding values differ, the
    function returns False.

    """
    if left is right:  # if they are the same object, return True
        return True

    # Extra keys
    keys = set(left).union(right)

    # If the keys are not the same on both sides, return False
    if not (len(keys) == len(left) == len(right)):
        return False

    is_equal = True  # Flag to check if all keys are equal, optimist
    while is_equal and keys:  # Loop until all keys are equal
        key = keys.pop()
        left_value, right_value = left[key], right[key]

        if type(left_value) is not type(right_value):
            is_equal = False

        elif isinstance(left_value, Mapping):
            is_equal = dict_allclose(
                left_value,
                right_value,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )

        elif isinstance(left_value, np.ndarray) and issubclass(
            left_value.dtype.type, _INEXACT_TYPES
        ):
            is_equal = np.allclose(
                left_value,
                right_value,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )

        else:
            is_equal = np.array_equal(left_value, right_value)

    return is_equal
