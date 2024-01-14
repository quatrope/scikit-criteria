#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Public testing utility functions.

This module exposes "assert" functions which facilitate the comparison in a
testing environment of objects created in skcriteria.

The functionalities are extensions of those present in "pandas.testing" and
"numpy.testing".

"""

# =============================================================================
# IMPORTS
# =============================================================================

from .utils import hidden

with hidden():
    import numpy as np
    import numpy.testing as npt

    import pandas.testing as pdt

    from .agg import ResultABC
    from .core import DecisionMatrix
    from .cmp import RanksComparator


# =============================================================================
# CONSTANTS
# =============================================================================


def _assert(cond, err_msg):
    """Asserts that a condition is true, otherwise raises an AssertionError \
    with a specified error message.

    This function exists to prevent asserts from being turned off with a
    "python -O."

    Parameters
    ----------
    cond : bool
        The condition to be evaluated.
    err_msg : str
        The error message to be raised if the condition is false.

    """
    if not cond:
        raise AssertionError(err_msg)


def assert_dmatrix_equals(left, right, **diff_kws):
    """Asserts that two instances of the DecisionMatrix class are equal."""
    _assert(
        isinstance(left, DecisionMatrix),
        f"'left' is not a DecisionMatrix instance. Found {type(left)!r}",
    )

    diff = left.diff(right, **diff_kws)

    if not diff.has_differences:
        return

    _assert(
        diff.right_type is DecisionMatrix,
        f"'right' is not a DecisionMatrix instance. Found {type(right)!r}",
    )

    assert ("shape" not in diff.members_diff, "'shape' are not equal")
    _assert("criteria" not in diff.members_diff, "'criteria' are not equal")
    _assert(
        "alternatives" not in diff.members_diff, "'alternatives' are not equal"
    )
    _assert(
        "objectives" not in diff.members_diff, "'objectives' are not equal"
    )
    _assert("weights" not in diff.members_diff, "'weights' are not equal")
    _assert("matrix" not in diff.members_diff, "'matrix' are not equal")
    _assert("dtypes" not in diff.members_diff, "'dtypes' are not equal")


def assert_result_equals(left, right):
    """Asserts that two instances of the Result class are equals.

    Parameters
    ----------
    left : Result
        The first Result instance for comparison.
    right : Result
        The second Result instance for comparison.

    Raises
    ------
    AssertionError
        If any of the specified attributes of the two Result instances are
        not equal.

    Notes
    -----
    This function uses NumPy testing utilities for array and string
    comparisons.

    Example
    -------
    >>> assert_result_equals(result1, result2)
    """
    assert isinstance(
        left, ResultABC
    ), f"'left' is not a ResultABC instance. Found {type(left)!r}"
    assert isinstance(
        right, ResultABC
    ), f"'right' is not a ResultABC instance. Found {type(right)!r}"

    # if the objects are the same, no need to run the test
    if left is right:
        return

    t_left, t_right = type(left), type(right)
    assert (
        t_left is t_right
    ), f"Type mismatch: Expected instances of {t_left!r}, but got {t_right!r}."

    # Check equality of alternatives and criteria arrays
    npt.assert_array_equal(
        np.asarray(left.alternatives),
        np.asarray(right.alternatives),
        err_msg="Alternatives are not equal",
    )

    # Check equality of method attribute (string comparison)
    assert (
        left.method == right.method
    ), f"Method mismatch: Expected {left.method!r}, but got {right.method!r}."

    # Check equality of the alternatives arrays
    npt.assert_array_equal(
        np.asarray(left.values),
        np.asarray(right.values),
        err_msg="Values are not equal",
    )

    # Check equality of extra attribute (Bunch comparison)
    npt.assert_equal(
        dict(left.e_), dict(right.e_), err_msg="Extras are not equal"
    )


def assert_rcmp_equals(left, right):
    """Asserts that two instances of the RanksComparator class are equal.

    Parameters
    ----------
    left : RanksComparator
        The first RanksComparator instance for comparison.
    right : RanksComparator
        The second RanksComparator instance for comparison.

    Raises
    ------
    AssertionError
        If any of the specified attributes of the two RanksComparator instances
        are not equal.

    Notes
    -----
    This function relies on the assert_result_equals function for comparing
    individual Result instances.

    Example
    -------
    >>> assert_rcmp_equals(rcmp1, rcmp2)
    """
    assert isinstance(
        left, RanksComparator
    ), f"'left' is not a RanksComparator instance. Found {type(left)!r}"
    assert isinstance(
        right, RanksComparator
    ), f"'right' is not a RanksComparator instance. Found {type(right)!r}"

    # if the objects are the same, no need to run the test
    if left is right:
        return

    llen, rlen = len(left), len(right)
    assert len(left) == len(
        right
    ), f"RanksComparator instances have different lengths: {llen} != {rlen}"

    for idx, (lrank, rrank) in enumerate(zip(left, right)):
        try:
            assert_result_equals(lrank, rrank)
        except AssertionError as err:
            raise AssertionError(f"Mismatch at index {idx}") from err
