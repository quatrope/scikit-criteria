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

import numpy as np

import pytest

from skcriteria import base, data


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


def test_repr_no_params():
    class Foo(base.SKCBaseDecisionMaker):
        _skcriteria_dm_type = "foo"

    foo = Foo()

    assert repr(foo) == "Foo()"


# =============================================================================
# TRANSFORMER
# =============================================================================


def test_not_redefined_SKCTransformerMixin():
    class Foo(base.SKCTransformerMixin, base.SKCBaseDecisionMaker):
        pass

    with pytest.raises(TypeError):
        Foo()


# =============================================================================
# MATRIX AND WEIGHT TRANSFORMER
# =============================================================================


def test_transform_data_not_implemented_SKCMatrixAndWeightTransformerMixin(
    decision_matrix,
):
    class Foo(base.SKCTransformerMixin, base.SKCBaseDecisionMaker):
        def _transform_data(self, **kwargs) -> dict:
            return super()._transform_data(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_not_redefined_SKCMatrixAndWeightTransformerMixin():
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        pass

    with pytest.raises(TypeError):
        Foo("matrix")

    with pytest.raises(TypeError):
        Foo("weights")

    with pytest.raises(TypeError):
        Foo("both")


def test_bad_normalize_for_SKCMatrixAndWeightTransformerMixin():
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        def _transform_matrix(self, matrix):
            ...

        def _transform_weights(self, weights):
            ...

    with pytest.raises(ValueError):
        Foo("mtx")


def test_transform_weights_not_implemented_SKCMatrixAndWeightTransformerMixin(
    decision_matrix,
):
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        def _transform_matrix(self, matrix):
            super()._transform_matrix(matrix)

        def _transform_weights(self, weights):
            return weights

    transformer = Foo("matrix")
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_transform_weight_not_implemented_SKCMatrixAndWeightTransformerMixin(
    decision_matrix,
):
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
        def _transform_matrix(self, matrix):
            return matrix

        def _transform_weights(self, weights):
            super()._transform_weights(weights)

    transformer = Foo("weights")
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_SKCMatrixAndWeightTransformerMixin_target():
    class Foo(
        base.SKCMatrixAndWeightTransformerMixin, base.SKCBaseDecisionMaker
    ):
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


# =============================================================================
# MATRIX AND WEIGHT TRANSFORMER
# =============================================================================


def test_weight_matrix_not_implemented_SKCWeighterMixin(decision_matrix):
    class Foo(base.SKCWeighterMixin, base.SKCBaseDecisionMaker):
        def _weight_matrix(self, matrix) -> dict:
            return super()._weight_matrix(matrix)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_not_redefined_SKCWeighterMixin():
    class Foo(base.SKCWeighterMixin, base.SKCBaseDecisionMaker):
        pass

    with pytest.raises(TypeError):
        Foo()


def test_flow_SKCWeighterMixin(decision_matrix):

    dm = decision_matrix(seed=42)
    expected_weights = np.ones(dm.matrix.shape[1]) * 42

    class Foo(base.SKCWeighterMixin, base.SKCBaseDecisionMaker):
        def _weight_matrix(self, matrix) -> dict:
            return expected_weights

    transformer = Foo()

    expected = data.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=expected_weights,
        dtypes=dm.dtypes,
        anames=dm.anames,
        cnames=dm.cnames,
    )

    result = transformer.transform(dm)

    assert result.equals(expected)
