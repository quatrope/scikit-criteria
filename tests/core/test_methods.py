#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.methods

"""


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

    assert repr(foo) == "Foo(faa=1, foo=2)"


def test_SKCMethodABC_repr_no_params():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"
        _skcriteria_parameters = []

    foo = Foo()

    assert repr(foo) == "Foo()"


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


# =============================================================================
# TRANSFORMER
# =============================================================================


def test_SKCTransformerMixin_not_redefined_abc_methods():
    class Foo(methods.SKCTransformerABC):
        _skcriteria_parameters = []

    with pytest.raises(TypeError):
        Foo()


# =============================================================================
# MATRIX AND WEIGHT TRANSFORMER
# =============================================================================


def test_SKCMatrixAndWeightTransformerMixin_transform_data_not_implemented(
    decision_matrix,
):
    class Foo(methods.SKCTransformerABC):
        _skcriteria_parameters = []

        def _transform_data(self, **kwargs):
            return super()._transform_data(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerMixin_not_redefined_abc_methods():
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        pass

    with pytest.raises(TypeError):
        Foo("matrix")

    with pytest.raises(TypeError):
        Foo("weights")

    with pytest.raises(TypeError):
        Foo("both")


def test_SKCMatrixAndWeightTransformerMixin_bad_normalize_for():
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            ...

        def _transform_weights(self, weights):
            ...

    with pytest.raises(ValueError):
        Foo("mtx")


def test_SKCMatrixAndWeightTransformerMixin_transform_weights_not_implemented(
    decision_matrix,
):
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            super()._transform_matrix(matrix)

        def _transform_weights(self, weights):
            return weights

    transformer = Foo("matrix")
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerMixin_transform_weight_not_implemented(
    decision_matrix,
):
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            return matrix

        def _transform_weights(self, weights):
            super()._transform_weights(weights)

    transformer = Foo("weights")
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerMixin_target():
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            ...

        def _transform_weights(self, weights):
            ...

    foo = Foo("matrix")
    assert foo.target == Foo._TARGET_MATRIX

    foo = Foo("weights")
    assert foo.target == Foo._TARGET_WEIGHTS

    foo = Foo("matrix")
    assert foo.target == Foo._TARGET_MATRIX

    foo = Foo("both")
    assert foo.target == Foo._TARGET_BOTH
