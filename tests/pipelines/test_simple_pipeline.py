#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.pipelines"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria import pipelines
from skcriteria.agg.topsis import TOPSIS
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import StandarScaler
from skcriteria.preprocessing.weighters import CRITIC

# =============================================================================
# TESTS
# =============================================================================


def test_pipeline_mkpipe(decision_matrix):
    dm = decision_matrix(seed=42)

    steps = [
        InvertMinimize(),
        StandarScaler(target="matrix"),
        CRITIC(correlation="spearman"),
        CRITIC(),
        TOPSIS(),
    ]

    expected = dm
    for step in steps[:-1]:
        expected = step.transform(expected)
    expected = steps[-1].evaluate(expected)

    pipe = pipelines.mkpipe(*steps)
    result = pipe.evaluate(dm)

    assert result.values_equals(expected)
    assert len(pipe) == len(steps)
    assert steps == [s for _, s in pipe.steps]
    for s in pipe.named_steps.values():
        assert s in steps


def test_pipeline_slicing():
    steps = [
        InvertMinimize(),
        StandarScaler(target="matrix"),
        CRITIC(correlation="spearman"),
        CRITIC(),
        TOPSIS(),
    ]

    pipe = pipelines.mkpipe(*steps)

    for idx, step in enumerate(steps):
        assert pipe[idx] == step

    for name, step in pipe.named_steps.items():
        assert pipe[name] == step

    assert [s for _, s in pipe[2:].steps] == steps[2:]

    with pytest.raises(ValueError):
        pipe[::2]

    with pytest.raises(KeyError):
        pipe[None]


def test_pipeline_not_transformer_fail():
    steps = [TOPSIS(), TOPSIS()]
    with pytest.raises(TypeError):
        pipelines.mkpipe(*steps)


def test_pipeline_not_dmaker_fail():
    steps = [CRITIC(), CRITIC()]
    with pytest.raises(TypeError):
        pipelines.mkpipe(*steps)


def test_pipeline_not_steps():
    steps = []
    with pytest.raises(ValueError):
        pipelines.mkpipe(*steps)


def test_pipeline_name_not_str():
    with pytest.raises(TypeError):
        pipelines.SKCPipeline(steps=[(..., CRITIC()), ("final", TOPSIS())])
    with pytest.raises(TypeError):
        pipelines.SKCPipeline(steps=[("first", CRITIC()), (..., TOPSIS())])
