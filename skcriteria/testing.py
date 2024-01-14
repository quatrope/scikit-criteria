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
    """Asserts that two DecisionMatrix objects are equal by comparing \
    their attributes with some tolerance.

    Parameters
    ----------
    left : DecisionMatrix
        The first DecisionMatrix object to compare.
    right : DecisionMatrix
        The second DecisionMatrix object to compare.
    **diff_kws : dict
        Additional keyword arguments to pass to the `DecisionMatrix.diff`
        method.

    Raises
    ------
    AssertionError
        If the two DecisionMatrix objects are not equal.

    """
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

    _assert("shape" not in diff.members_diff, "'shape' are not equal")
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


def assert_result_equals(left, right, **diff_kws):
    """Asserts that two results objects are equal by comparing their \
    attributes with some tolerance.

    Parameters
    ----------
    left : ResultABC
        The left result to compare.
    right : ResultABC
        The right result to compare.
    **diff_kws : dict
        Optional keyword arguments to pass to the result `diff` method.

    Returns
    -------
    None

    Raises
    ------
    AssertionError if the two results are not equal.

    """
    _assert(
        isinstance(left, ResultABC),
        f"'left' is not a ResultABC instance. Found {type(left)!r}",
    )

    diff = left.diff(right, **diff_kws)

    if not diff.has_differences:
        return

    _assert(
        diff.different_types is False,
        f"'right' is not a ResultABC instance. Found {diff.right_type!r}",
    )
    _assert(
        "alternatives" not in diff.members_diff, "'alternatives' are not equal"
    )
    _assert(
        "method" not in diff.members_diff,
        f"'method' mismatch: Expected {left.method!r}, "
        f"but got {right.method!r}.",
    )

    _assert("values" not in diff.members_diff, "'values' are not equal")
    _assert("extra_" not in diff.members_diff, "'extra_' are not equal")


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
