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


def assert_dmatrix_equals(
    left,
    right,
    *,
    matrix_kws=None,
    objectives_kws=None,
    weights_kws=None,
):
    """Asserts that two instances of the DecisionMatrix class are equal.

    Parameters
    ----------
    left : DecisionMatrix
        The first DecisionMatrix instance for comparison.
    right : DecisionMatrix
        The second DecisionMatrix instance for comparison.
    matrix_kws : dict, optional
        Keyword arguments for pandas.testing.assert_frame_equal
        when comparing the matrix attribute. Default is an empty dict.
    objectives_kws : dict, optional
        Keyword arguments for pandas.testing.assert_series_equal
        when comparing the objectives attribute. Default is an empty dict.
    weights_kws : dict, optional
        Keyword arguments for pandas.testing.assert_series_equal
        when comparing the weights attribute. Default is an empty dict.

    Raises
    ------
    AssertionError
        If any of the specified attributes of the two DecisionMatrix instances
        are not equal.

    Notes
    -----
    This function uses NumPy and pandas testing utilities for array and
    DataFrame comparisons.

    By default, the comparison of the decision matrix (matrix attribute)
    does not check the columns dtypes (`check_dtype` is set to False) and
    allows for some tolerance (`check_exact` is set to False) when using the
    pandas testing function `assert_frame_equal`. Similarly, the comparison of
    objectives and weights using the pandas testing function
    `assert_series_equal` does not check the data type (`check_dtype` is set to
    False) and allows for some tolerance (`check_exact` is set to False) by
    default.

    Example
    -------
    >>> assert_dmatrix_equals(
    ...     matrix1, matrix2,
    ...     matrix_kws={'check_dtype': True},
    ...     weights_kws={'check_exact': True})

    """
    assert isinstance(
        left, DecisionMatrix
    ), f"'left' is not a DecisionMatrix instance. Found {type(left)!r}"
    assert isinstance(
        right, DecisionMatrix
    ), f"'right' is not a DecisionMatrix instance. Found {type(right)!r}"

    # Check equality of alternatives and criteria arrays
    npt.assert_array_equal(
        np.asarray(left.alternatives),
        np.asarray(right.alternatives),
        err_msg="Alternatives are not equal",
    )
    npt.assert_array_equal(
        np.asarray(left.criteria),
        np.asarray(right.criteria),
        err_msg="Criteria are not equal",
    )

    # setup the defaults
    matrix_kws = {} if matrix_kws is None else matrix_kws
    objectives_kws = {} if objectives_kws is None else objectives_kws
    weights_kws = {} if weights_kws is None else weights_kws

    # Check equality of decision matrix DataFrame
    matrix_kws.setdefault("check_dtype", False)
    matrix_kws.setdefault("check_exact", False),
    pdt.assert_frame_equal(
        left.matrix,
        right.matrix,
        obj=DecisionMatrix.__name__,
        **matrix_kws,
    )

    # Check equality of objectives and weights Series
    objectives_kws.setdefault("check_exact", False),
    objectives_kws.setdefault("check_dtype", False),
    pdt.assert_series_equal(
        left.objectives, right.objectives, obj="Objectives", **objectives_kws
    )

    objectives_kws.setdefault("check_dtype", False),
    objectives_kws.setdefault("check_exact", False),
    pdt.assert_series_equal(
        left.weights, right.weights, obj="Weights", **weights_kws
    )


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
    npt.assert_equal(left.extra_, right.extra_, err_msg="Extras are not equal")


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

    llen, rlen = len(left), len(right)
    assert len(left) == len(
        right
    ), f"RanksComparator instances have different lengths: {llen} != {rlen}"

    for idx, (lrank, rrank) in enumerate(zip(left, right)):
        try:
            assert_result_equals(lrank, rrank)
        except AssertionError as err:
            raise AssertionError(f"Mismatch at index {idx}") from err
