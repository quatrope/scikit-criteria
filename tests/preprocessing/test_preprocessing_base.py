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

from skcriteria.preprocessing import (
    SKCMatrixAndWeightTransformerABC,
    SKCTransformerABC,
)


# =============================================================================
# TRANSFORMER
# =============================================================================


def test_SKCTransformerABC_not_redefined_abc_methods():
    class Foo(SKCTransformerABC):
        _skcriteria_parameters = []

    with pytest.raises(TypeError):
        Foo()


# =============================================================================
# MATRIX AND WEIGHT TRANSFORMER
# =============================================================================


def test_SKCMatrixAndWeightTransformerABC_transform_data_not_implemented(
    decision_matrix,
):
    class Foo(SKCTransformerABC):
        _skcriteria_parameters = []

        def _transform_data(self, **kwargs):
            return super()._transform_data(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerABC_not_redefined_abc_methods():
    class Foo(SKCMatrixAndWeightTransformerABC):
        pass

    with pytest.raises(TypeError):
        Foo("matrix")

    with pytest.raises(TypeError):
        Foo("weights")

    with pytest.raises(TypeError):
        Foo("both")


def test_SKCMatrixAndWeightTransformerABC_bad_normalize_for():
    class Foo(SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix): ...

        def _transform_weights(self, weights): ...

    with pytest.raises(ValueError):
        Foo("mtx")


def test_SKCMatrixAndWeightTransformerABC_transform_weights_not_implemented(
    decision_matrix,
):
    class Foo(SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            super()._transform_matrix(matrix)

        def _transform_weights(self, weights):
            return weights

    transformer = Foo("matrix")
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerABC_transform_weight_not_implemented(
    decision_matrix,
):
    class Foo(SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            return matrix

        def _transform_weights(self, weights):
            super()._transform_weights(weights)

    transformer = Foo("weights")
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerABC_target():
    class Foo(SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix): ...

        def _transform_weights(self, weights): ...

    foo = Foo("matrix")
    assert foo.target == Foo._TARGET_MATRIX

    foo = Foo("weights")
    assert foo.target == Foo._TARGET_WEIGHTS

    foo = Foo("matrix")
    assert foo.target == Foo._TARGET_MATRIX

    foo = Foo("both")
    assert foo.target == Foo._TARGET_BOTH
