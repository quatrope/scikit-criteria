#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.extend"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria import extend as ext
from skcriteria.pipelines import mkpipe


# =============================================================================
# RANDOM TEST
# =============================================================================


def test_mkagg(decision_matrix):
    dm = decision_matrix(seed=42, min_alternatives=10)
    values = np.arange(len(dm)) + 1
    np.random.shuffle(values)

    @ext.mkagg
    def MyAgg(**kwargs):
        return values, {}

    dec = MyAgg()

    assert repr(dec) == "<MyAgg []>"

    rank = dec.evaluate(dm)

    assert rank.method == "MyAgg"
    np.testing.assert_array_equal(rank.alternatives, dm.alternatives)
    np.testing.assert_array_equal(rank.values, values)
    assert rank.extra_ == {}


def test_mktransformer(decision_matrix):
    dm = decision_matrix(seed=42, min_alternatives=10)

    @ext.mktransformer
    def MyTransformer(matrix, weights, hparams, **kwargs):
        return {"matrix": matrix + 1, "weights": weights + 1}

    dec = MyTransformer()

    assert repr(dec) == "<MyTransformer []>"

    dmt = dec.transform(dm)

    np.testing.assert_array_equal(dmt.matrix, dm.matrix + 1)
    np.testing.assert_array_equal(dmt.weights, dm.weights + 1)


def test_mkagg_CapWords():
    with pytest.warns(ext.NonStandardNameWarning):

        @ext.mkagg
        def agg(**kwargs):
            pass


def test_mktransformer_CapWords():
    with pytest.warns(ext.NonStandardNameWarning):

        @ext.mktransformer
        def transformer(**kwargs):
            pass


def test_mkagg_invalid_parameter():
    with pytest.raises(TypeError):

        @ext.mkagg
        def Agg(foo):
            pass


def test_mktransformer_invalid_parameter():
    with pytest.raises(TypeError):

        @ext.mktransformer
        def Transformer(foo):
            pass


def test_mkagg_missing_parameter():
    with pytest.raises(TypeError):

        @ext.mkagg
        def Agg(matrix, objectives, weights, dtypes, criteria):
            pass


def test_mktransformer_missing_parameter():
    with pytest.raises(TypeError):

        @ext.mktransformer
        def Transformer(matrix, objectives, weights, dtypes, criteria):
            pass


def test_mkagg_invalid_argument():
    @ext.mkagg
    def Agg(**kwargs):
        pass

    with pytest.raises(TypeError):
        Agg(x=1)


def test_mktransformer_invalid_argument():
    @ext.mktransformer
    def Transformer(**kwargs):
        pass

    with pytest.raises(TypeError):
        Transformer(x=1)


def test_mkagg_in_pipeline(decision_matrix):
    dm = decision_matrix(seed=42, min_alternatives=10)

    # TRANSFORMER =============================================================

    @ext.mktransformer
    def MyTransformer(matrix, weights, hparams, **kwargs):
        return {"matrix": matrix + 1, "weights": weights + 1}

    # AGG =====================================================================
    values = np.arange(len(dm)) + 1
    np.random.shuffle(values)

    @ext.mkagg
    def MyAgg(**kwargs):
        return values, {}

    # PIPE ====================================================================
    pipe = mkpipe(MyTransformer(), MyAgg())

    assert repr(pipe) == (
        "<SKCPipeline "
        "[steps=[('mytransformer', <MyTransformer []>), "
        "('myagg', <MyAgg []>)]]>"
    )

    rank = pipe.evaluate(dm)
    assert rank.method == "MyAgg"
    np.testing.assert_array_equal(rank.alternatives, dm.alternatives)
    np.testing.assert_array_equal(rank.values, values)
    assert rank.extra_ == {}

    dmt = pipe.transform(dm)
    np.testing.assert_array_equal(dmt.matrix, dm.matrix + 1)
    np.testing.assert_array_equal(dmt.weights, dm.weights + 1)
