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

import numpy as np

import pytest

from skcriteria.core import data, methods


# =============================================================================
# TESTS
# =============================================================================


def test_SKCMethodABC_no__skcriteria_dm_type():

    with pytest.raises(TypeError):

        class Foo(methods.SKCMethodABC):
            pass


def test_SKCMethodABC_varargs_for__skcriteria_parameters_inspection():

    with pytest.raises(TypeError):

        class Foo(methods.SKCMethodABC):
            _skcriteria_dm_type = "foo"

            def __init__(self, **kwargs):
                pass


def test_SKCMethodABC_repr():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert repr(foo) == "Foo(faa=1, foo=2)"


def test_SKCMethodABC_repr_no_params():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"

    foo = Foo()

    assert repr(foo) == "Foo()"


def test_SKCMethodABC_no_params():
    class Foo(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"

    assert Foo._skcriteria_parameters == set()


def test_SKCMethodABC_alreadydefines__skcriteria_parameters():
    class Base(methods.SKCMethodABC):
        _skcriteria_dm_type = "foo"

    class Foo(Base):
        def __init__(self, x):
            pass

    assert Foo._skcriteria_parameters == {"x"}


# =============================================================================
# TRANSFORMER
# =============================================================================


def test_not_redefined_SKCTransformerMixin():
    class Foo(methods.SKCTransformerABC):
        pass

    with pytest.raises(TypeError):
        Foo()


# =============================================================================
# MATRIX AND WEIGHT TRANSFORMER
# =============================================================================


def test_transform_data_not_implemented_SKCMatrixAndWeightTransformerMixin(
    decision_matrix,
):
    class Foo(methods.SKCTransformerABC):
        def _transform_data(self, **kwargs):
            return super()._transform_data(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_not_redefined_SKCMatrixAndWeightTransformerMixin():
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        pass

    with pytest.raises(TypeError):
        Foo("matrix")

    with pytest.raises(TypeError):
        Foo("weights")

    with pytest.raises(TypeError):
        Foo("both")


def test_bad_normalize_for_SKCMatrixAndWeightTransformerMixin():
    class Foo(methods.SKCMatrixAndWeightTransformerABC):
        def _transform_matrix(self, matrix):
            ...

        def _transform_weights(self, weights):
            ...

    with pytest.raises(ValueError):
        Foo("mtx")


def test_transform_weights_not_implemented_SKCMatrixAndWeightTransformerMixin(
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


def test_transform_weight_not_implemented_SKCMatrixAndWeightTransformerMixin(
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


# =============================================================================
# MATRIX AND WEIGHT TRANSFORMER
# =============================================================================


def test_weight_matrix_not_implemented_SKCWeighterMixin(decision_matrix):
    class Foo(methods.SKCWeighterABC):
        def _weight_matrix(self, **kwargs):
            return super()._weight_matrix(**kwargs)

    transformer = Foo()
    dm = decision_matrix(seed=42)

    with pytest.raises(NotImplementedError):
        transformer.transform(dm)


def test_not_redefined_SKCWeighterMixin():
    class Foo(methods.SKCWeighterABC):
        pass

    with pytest.raises(TypeError):
        Foo()


def test_flow_SKCWeighterMixin(decision_matrix):

    dm = decision_matrix(seed=42)
    expected_weights = np.ones(dm.matrix.shape[1]) * 42

    class Foo(methods.SKCWeighterABC):
        def _weight_matrix(self, matrix, **kwargs):
            return expected_weights

    transformer = Foo()

    expected = data.mkdm(
        matrix=dm.matrix,
        objectives=dm.objectives,
        weights=expected_weights,
        dtypes=dm.dtypes,
        alternatives=dm.alternatives,
        criteria=dm.criteria,
    )

    result = transformer.transform(dm)

    assert result.equals(expected)


# =============================================================================
# SKCDecisionMakerABC
# =============================================================================


def test_flow_SKCDecisionMakerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(methods.SKCDecisionMakerABC):
        def _evaluate_data(self, alternatives, **kwargs):
            return np.arange(len(alternatives)) + 1, {}

        def _make_result(self, alternatives, values, extra):
            return {
                "alternatives": alternatives,
                "rank": values,
                "extra": extra,
            }

    ranker = Foo()

    result = ranker.evaluate(dm)

    assert np.all(result["alternatives"] == dm.alternatives)
    assert np.all(result["rank"] == np.arange(len(dm.alternatives)) + 1)
    assert result["extra"] == {}


@pytest.mark.parametrize("not_redefine", ["_evaluate_data", "_make_result"])
def test_not_redefined_SKCDecisionMakerMixin(not_redefine):
    content = {}
    for method_name in ["_evaluate_data", "_make_result", "_validate_data"]:
        if method_name != not_redefine:
            content[method_name] = lambda **kws: None

    Foo = type("Foo", (methods.SKCDecisionMakerABC,), content)

    with pytest.raises(TypeError):
        Foo()


def test_evaluate_data_not_implemented_SKCDecisionMakerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(methods.SKCDecisionMakerABC):
        def _evaluate_data(self, **kwargs):
            super()._evaluate_data(**kwargs)

        def _make_result(self, alternatives, values, extra):
            return {
                "alternatives": alternatives,
                "rank": values,
                "extra": extra,
            }

    ranker = Foo()

    with pytest.raises(NotImplementedError):
        ranker.evaluate(dm)


def test_make_result_not_implemented_SKCDecisionMakerMixin(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(methods.SKCDecisionMakerABC):
        def _evaluate_data(self, alternatives, **kwargs):
            return np.arange(len(alternatives)) + 1, {}

        def _make_result(self, **kwargs):
            super()._make_result(**kwargs)

    ranker = Foo()

    with pytest.raises(NotImplementedError):
        ranker.evaluate(dm)


# subclass testing


def _get_subclasses(cls):

    if (
        cls._skcriteria_dm_type not in (None, "pipeline")
        and not cls.__abstractmethods__
    ):
        yield cls

    for subc in cls.__subclasses__():
        for subsub in _get_subclasses(subc):
            yield subsub


@pytest.mark.run(order=-1)
def test_SLCMethodABC_concrete_subclass_copy():

    extra_parameters_by_type = {
        methods.SKCMatrixAndWeightTransformerABC: {"target": "both"}
    }

    for scls in _get_subclasses(methods.SKCMethodABC):
        kwargs = {}
        for cls, extra_params in extra_parameters_by_type.items():
            if issubclass(scls, cls):
                kwargs.update(extra_params)

        original = scls(**kwargs)
        copy = original.copy()

        assert (
            original.get_parameters() == copy.get_parameters()
        ), f"'{scls.__qualname__}' instance not correctly copied."
