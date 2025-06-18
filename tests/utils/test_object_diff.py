#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.object_diff"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria.utils import object_diff


# =============================================================================
# The tests
# =============================================================================


def test_MISSING():
    assert object_diff.MISSING is object_diff._Missing()
    assert repr(object_diff.MISSING) == "<MISSING>"


def test_diff():
    class SomeClass:
        def __init__(self, **kws):
            self.__dict__.update(kws)

    obj_a = SomeClass(a=1, b=2)
    obj_b = SomeClass(a=1, b=3, c=4)
    result = object_diff.diff(
        obj_a, obj_b, a=np.equal, b=np.equal, c=np.equal, d=np.equal
    )

    assert result.left_type is SomeClass
    assert result.right_type is SomeClass
    assert result.different_types is False

    assert result.has_differences

    assert tuple(sorted(result.members_diff)) == ("b", "c")
    assert result.members_diff["b"] == (2, 3)
    assert result.members_diff["c"] == (object_diff.MISSING, 4)

    expected_repr = (
        "<Difference "
        "has_differences=True "
        "different_types=False "
        "members_diff=('b', 'c')>"
    )
    assert repr(result) == expected_repr

    # reverting the order of the arguments ====================================

    result = object_diff.diff(
        obj_b, obj_a, a=np.equal, b=np.equal, c=np.equal, d=np.equal
    )

    assert result.left_type is SomeClass
    assert result.right_type is SomeClass
    assert result.different_types is False

    assert result.has_differences

    assert tuple(sorted(result.members_diff)) == ("b", "c")
    assert result.members_diff["b"] == (3, 2)
    assert result.members_diff["c"] == (4, object_diff.MISSING)

    expected_repr = (
        "<Difference "
        "has_differences=True "
        "different_types=False "
        "members_diff=('b', 'c')>"
    )
    assert repr(result) == expected_repr


def test_diff_different_types():
    class SomeClass:
        def __init__(self, **kws):
            self.__dict__.update(kws)

    obj_a = SomeClass(a=1, b=2)
    obj_b = 1
    result = object_diff.diff(obj_a, obj_b)

    assert result.left_type is SomeClass
    assert result.right_type is int
    assert result.different_types
    assert result.has_differences


def test_diff_same_object():
    obj = 1
    result = object_diff.diff(obj, obj)

    assert result.different_types is False
    assert result.has_differences is False


def test_diff_equal_object():
    class SomeClass:
        def __init__(self, **kws):
            self.__dict__.update(kws)

    obj_a = SomeClass(a=1, b=2, c=3)
    obj_b = SomeClass(a=1, b=2, c=3)
    result = object_diff.diff(
        obj_a, obj_b, a=np.equal, b=np.equal, c=np.equal, d=np.equal
    )

    assert result.different_types is False
    assert result.has_differences is False


def test_DiffEqualityMixin():
    class SomeClass(object_diff.DiffEqualityMixin):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def diff(
            self,
            other,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=True,
            check_dtypes=False,
        ):
            return object_diff.diff(
                self, other, a=np.equal, b=np.equal, c=np.equal
            )

    obj_a = SomeClass(a=1, b=2, d=5)
    obj_b = SomeClass(a=1, b=3, c=4, d=6)

    result = obj_a.diff(obj_b)

    assert result.left_type is SomeClass
    assert result.right_type is SomeClass
    assert result.different_types is False

    assert result.has_differences

    assert tuple(sorted(result.members_diff)) == ("b", "c")
    assert result.members_diff["b"] == (2, 3)
    assert result.members_diff["c"] == (object_diff.MISSING, 4)

    expected_repr = (
        "<Difference "
        "has_differences=True "
        "different_types=False "
        "members_diff=('b', 'c')>"
    )
    assert repr(result) == expected_repr

    # check the aequals method
    assert (obj_a == obj_b) is False
    assert obj_a != obj_b
    assert obj_a.equals(obj_b) is False
    assert obj_a.aequals(obj_b) is False

    # reverting the order of the arguments ====================================

    result = obj_b.diff(obj_a)

    assert result.left_type is SomeClass
    assert result.right_type is SomeClass
    assert result.different_types is False

    assert result.has_differences

    assert tuple(sorted(result.members_diff)) == ("b", "c")
    assert result.members_diff["b"] == (3, 2)
    assert result.members_diff["c"] == (4, object_diff.MISSING)

    expected_repr = (
        "<Difference "
        "has_differences=True "
        "different_types=False "
        "members_diff=('b', 'c')>"
    )
    assert repr(result) == expected_repr

    # check the aequals method
    assert (obj_a == obj_b) is False
    assert obj_a != obj_b
    assert obj_a.equals(obj_b) is False
    assert obj_a.aequals(obj_b) is False


def test_DiffEqualityMixin_diff_not_implemented():
    class SomeClass(object_diff.DiffEqualityMixin):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def diff(
            self,
            other,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=True,
            check_dtypes=False,
        ):
            return super().diff(
                other,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                check_dtypes=check_dtypes,
            )

    obj_a = SomeClass(a=1, b=2, d=5)

    with pytest.raises(NotImplementedError):
        obj_a.diff(1)


def test_DiffEqualityMixin_invalid_diff_parameters():
    with pytest.raises(TypeError):

        class SomeClass(object_diff.DiffEqualityMixin):
            def diff(self, other):
                pass
