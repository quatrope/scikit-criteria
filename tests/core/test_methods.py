#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.methods"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria.core import methods


# =============================================================================
# TESTS
# =============================================================================


def test_SKCMethodABC_no__skcriteria_dm_type():
    with pytest.raises(TypeError):

        class Foo(methods.SKCMethodABC):
            pass


def test_SKCMethodABC_no__skcriteria_parameters():
    with pytest.raises(TypeError):

        class Foo(methods.SKCMethodABC):
            _skcriteria_dm_type = "foo"

            def __init__(self, **kwargs):
                pass


def test_SKCMethodABC_repr():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert repr(foo) == "<Foo [faa=1, foo=2]>"


def test_SKCMethodABC_repr_no_params():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = []

    foo = Foo()

    assert repr(foo) == "<Foo []>"


def test_SKCMethodABC_no_params():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = []

    assert Foo._skcriteria_parameters == frozenset()


def test_SKCMethodABC_already_defined__skcriteria_parameters():
    class Base(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = ["x"]

        def __init__(self, x):
            pass

    class Foo(Base):
        def __init__(self, x):
            pass

    assert Foo._skcriteria_parameters == {"x"}


def test_SKCMethodABC_params_in_init():
    class Base(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = ["x"]

        def __init__(self, **kwargs):
            pass

    with pytest.raises(TypeError):

        class Foo(Base):
            def __init__(self):
                pass


def test_SKCMethodABC_get_parameters():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert foo.get_parameters() == {"foo": 2, "faa": 1}


def test_SKCMethodABC_copy():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    original = Foo(foo=2, faa=1)
    copy = original.copy()

    assert original.get_parameters() == copy.get_parameters()

    with pytest.deprecated_call():
        original.copy(foo=100)


def test_SKCMethodABC_replace():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    original = Foo(foo=2, faa=1)
    copy = original.replace(foo=100)

    assert original.get_parameters() != copy.get_parameters()
    assert copy.get_parameters() == {"faa": 1, "foo": 100}
