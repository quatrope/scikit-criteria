#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.object_diff

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


from skcriteria.utils import object_diff


# =============================================================================
# The tests
# =============================================================================


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
