#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
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
    # Check if left is a DecisionMatrix
    _assert(
        isinstance(left, DecisionMatrix),
        f"'left' is not a DecisionMatrix instance. Found {type(left)!r}",
    )

    # Check if left and right are equal
    diff = left.diff(right, **diff_kws)
    if not diff.has_differences:  # if there are no differences end the test
        return

    # Check if right is a DecisionMatrix
    _assert(
        diff.right_type is DecisionMatrix,
        f"'right' is not a DecisionMatrix instance. Found {type(right)!r}",
    )

    # Check wich member are different
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
    left : skcriteria.agg.ResultABC
        The left result to compare.
    right : skcriteria.agg.ResultABC
        The right result to compare.
    **diff_kws : dict
        Optional keyword arguments to pass to the result `diff` method.

    Raises
    ------
    AssertionError if the two results are not equal.

    """
    # Check if left is a ResultABC
    _assert(
        isinstance(left, ResultABC),
        f"'left' is not a ResultABC instance. Found {type(left)!r}",
    )

    # check if left and right are equal
    diff = left.diff(right, **diff_kws)
    if not diff.has_differences:  # if there are no differences end the test
        return

    # check if right is a ResultABC
    _assert(
        diff.different_types is False,
        f"'right' is not a ResultABC instance. Found {diff.right_type!r}",
    )

    # Check wich member are different
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


def assert_rcmp_equals(left, right, **diff_kws):
    """Asserts that the left and right RankComparator objects are equal \
    by comparing their attributes with some tolerance.

    Parameters
    ----------
    left : RanksComparator
        The left object to compare.
    right : Any
        The right object to compare.
    **diff_kws : keyword arguments
        Additional keyword arguments to pass to the `diff` method.

    Raises
    ------
    AssertionError
        If the left object is not an instance of RanksComparator.
    AssertionError
        If the right object is not an instance of RanksComparator.
    AssertionError
        If the left and right objects have different lengths.
    AssertionError
        If the ranks at any index of the left and right objects are not
        equal.

    """
    # check if left is a RanksComparator
    _assert(
        isinstance(left, RanksComparator),
        f"'left' is not a RanksComparator instance. Found {type(left)!r}",
    )

    # check if left and right has some difference
    diff = left.diff(right, **diff_kws)
    if not diff.has_differences:  # if there are no differences end the test
        return

    # check if right is a RanksComparator
    _assert(
        diff.different_types is False,
        "'right' is not a RanksComparator instance. "
        f"Found {diff.right_type!r}",
    )

    # check if left and right have the same length
    llen, rlen = len(left), len(right)
    _assert(
        llen == rlen,
        f"RanksComparator instances have different lengths: {llen} != {rlen}",
    )

    # check if left and right have the same ranks
    enum_zip_ranks = enumerate(zip(left.ranks, right.ranks))
    for idx, ((lrank_name, lrank), (rrank_name, rrank)) in enum_zip_ranks:
        try:
            _assert(lrank_name == rrank_name, "Name missmatch")
            assert_result_equals(lrank, rrank)
        except AssertionError as err:
            raise AssertionError(f"Mismatch at index {idx}") from err
