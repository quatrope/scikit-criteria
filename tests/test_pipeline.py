#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.pipelines

"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria import madm, pipeline, preprocessing

# =============================================================================
# TESTS
# =============================================================================


def test_pipeline_make_pipeline(decision_matrix):
    dm = decision_matrix(seed=42)

    steps = [
        preprocessing.MinimizeToMaximize(),
        preprocessing.StandarScaler(target="matrix"),
        preprocessing.Critic(correlation="spearman"),
        preprocessing.Critic(),
        madm.TOPSIS(),
    ]

    expected = dm
    for step in steps[:-1]:
        expected = step.transform(expected)
    expected = steps[-1].evaluate(expected)

    pipe = pipeline.make_pipeline(*steps)
    result = pipe.evaluate(dm)

    assert result.equals(expected)
    assert len(pipe) == len(steps)
    assert steps == [s for _, s in pipe.steps]
    for s in pipe.named_steps.values():
        assert s in steps


def test_pipeline_slicing():

    steps = [
        preprocessing.MinimizeToMaximize(),
        preprocessing.StandarScaler(target="matrix"),
        preprocessing.Critic(correlation="spearman"),
        preprocessing.Critic(),
        madm.TOPSIS(),
    ]

    pipe = pipeline.make_pipeline(*steps)

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
    steps = [madm.TOPSIS(), madm.TOPSIS()]
    with pytest.raises(TypeError):
        pipeline.make_pipeline(*steps)


def test_pipeline_not_dmaker_fail():
    steps = [preprocessing.Critic()]
    with pytest.raises(TypeError):
        pipeline.make_pipeline(*steps)


def test_pipeline_name_not_str():
    with pytest.raises(TypeError):
        pipeline.SKCPipeline(
            steps=[(..., preprocessing.Critic()), ("final", madm.TOPSIS())]
        )
    with pytest.raises(TypeError):
        pipeline.SKCPipeline(
            steps=[("first", preprocessing.Critic()), (..., madm.TOPSIS())]
        )
