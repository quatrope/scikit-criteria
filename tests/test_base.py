#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.base

"""


# =============================================================================
# IMPORTS
# =============================================================================


import pytest

from skcriteria import base


# =============================================================================
# TESTS
# =============================================================================


def test_no__skcriteria_dm_type():

    with pytest.raises(TypeError):

        class Foo(base.SKCBaseDecisionMaker):
            pass


def test_repr():
    class Foo(base.SKCBaseDecisionMaker):
        _skcriteria_dm_type = "foo"

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert repr(foo) == "Foo(faa=1, foo=2)"


def test_not_implemented_SKCTransformerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(base.SKCTransformerMixin, base.SKCBaseDecisionMaker):
        pass

    foo = Foo()

    with pytest.raises(NotImplementedError):
        foo.transform(dm)


def test_not_implemented_SKCMatrixAndWeightTransformerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        pass

    foo = Foo("matrix")

    with pytest.raises(NotImplementedError):
        foo.transform(dm)

    foo = Foo("weights")

    with pytest.raises(NotImplementedError):
        foo.transform(dm)

    foo = Foo("both")

    with pytest.raises(NotImplementedError):
        foo.transform(dm)


def test_bad_normalize_for_SKCMatrixAndWeightTransformerMixin():
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        pass

    with pytest.raises(ValueError):
        Foo("mtx")


def test_SKCMatrixAndWeightTransformerMixin_target():
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        pass

    foo = Foo("matrix")
    assert foo.target == Foo._TARGET_MATRIX

    foo = Foo("weights")
    assert foo.target == Foo._TARGET_WEIGHTS

    foo = Foo("matrix")
    assert foo.target == Foo._TARGET_MATRIX

    foo = Foo("both")
    assert foo.target == Foo._TARGET_BOTH
