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

        class Foo(base.BaseDecisionMaker):
            pass


def test_repr():
    class Foo(base.BaseDecisionMaker):
        _skcriteria_dm_type = "foo"

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert repr(foo) == "Foo(faa=1, foo=2)"


def test_not_implemented_NormalizerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(base.NormalizerMixin, base.BaseDecisionMaker):
        pass

    foo = Foo()

    with pytest.raises(NotImplementedError):
        foo.normalize(dm)


def test_not_implemented_MatrixAndWeightNormalizerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(base.MatrixAndWeightNormalizerMixin, base.BaseDecisionMaker):
        pass

    foo = Foo("matrix")

    with pytest.raises(NotImplementedError):
        foo.normalize(dm)

    foo = Foo("weights")

    with pytest.raises(NotImplementedError):
        foo.normalize(dm)


def test_bad_normalize_for_MatrixAndWeightNormalizerMixin():
    class Foo(base.MatrixAndWeightNormalizerMixin, base.BaseDecisionMaker):
        pass

    with pytest.raises(ValueError):
        Foo(normalize_for="mtx")


def test_MatrixAndWeightNormalizerMixin_normalize_for():
    class Foo(base.MatrixAndWeightNormalizerMixin, base.BaseDecisionMaker):
        pass

    foo = Foo("matrix")
    assert foo.normalize_for == Foo._FOR_MATRIX

    foo = Foo("weights")
    assert foo.normalize_for == Foo._FOR_WEIGHTS

    foo = Foo("matrix")
    assert foo.normalize_for == Foo._FOR_MATRIX

    foo = Foo("both")
    assert foo.normalize_for == Foo._FOR_BOTH
