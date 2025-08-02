#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# IMPORTS
# =============================================================================

import pytest

import skcriteria as skc
from skcriteria.agg import simple
from skcriteria.pipelines.combinatorial import (
    SKCCombinatorialPipeline,
    mkcombinatorial,
)
from skcriteria.preprocessing import invert_objectives, scalers
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.utils import Bunch


# =============================================================================
# TESTS
# =============================================================================


def test_SKCCombinatorialPipeline_creation():

    steps = [
        ("inverter", invert_objectives.InvertMinimize()),
        (
            "scaler",
            [
                scalers.SumScaler(target="matrix"),
                scalers.VectorScaler(target="matrix"),
            ],
        ),
        ("agg", simple.WeightedSumModel()),
    ]

    pipeline = SKCCombinatorialPipeline(steps)
    assert len(pipeline.pipelines) == 2


def test_SKCCombinatorialPipeline_evaluate():

    dm = skc.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[1, 1, 1],
    )

    steps = [
        ("inverter", InvertMinimize()),
        (
            "scaler",
            [
                scalers.SumScaler(target="matrix"),
                scalers.VectorScaler(target="matrix"),
            ],
        ),
        ("agg", simple.WeightedSumModel()),
    ]

    pipeline = SKCCombinatorialPipeline(steps)
    result = pipeline.evaluate(dm)

    assert len(result) == 2
    ranks_names = [r[0] for r in result.ranks]
    assert "InvertMinimize_SumScaler_WeightedSumModel" in ranks_names
    assert "InvertMinimize_VectorScaler_WeightedSumModel" in ranks_names


def test_SKCCombinatorialPipeline_transform():
    dm = skc.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[1, 1, 1],
    )

    steps = [
        ("inverter", InvertMinimize()),
        (
            "scaler",
            [
                scalers.SumScaler(target="matrix"),
                scalers.VectorScaler(target="matrix"),
            ],
        ),
        ("agg", simple.WeightedSumModel()),
    ]

    pipeline = SKCCombinatorialPipeline(steps)
    transformed_dms = pipeline.transform(dm)

    assert isinstance(transformed_dms, Bunch)
    assert len(transformed_dms) == 2

    transformed_dms_names = list(transformed_dms.keys())
    assert "InvertMinimize_SumScaler_WeightedSumModel" in transformed_dms_names
    assert (
        "InvertMinimize_VectorScaler_WeightedSumModel" in transformed_dms_names
    )


def test_SKCCombinatorialPipeline_invalid_steps():
    with pytest.raises(ValueError):
        SKCCombinatorialPipeline(
            [("scaler", scalers.SumScaler(target="matrix"))]
        )


def test_SKCCombinatorialPipeline_evaluate2():
    """Test the evaluate method of CombinatorialPipeline."""
    dm = skc.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[1, 1, 1],
    )

    steps = [
        ("inverter", invert_objectives.InvertMinimize()),
        (
            "scaler",
            [
                scalers.SumScaler(target="matrix"),
                scalers.VectorScaler(target="matrix"),
            ],
        ),
        ("agg", simple.WeightedSumModel()),
    ]

    pipeline = SKCCombinatorialPipeline(steps)

    # The evaluate method should return a RanksComparator object
    result = pipeline.evaluate(dm)

    # Check that the result has the expected number of ranks
    assert len(result) == 2


def test_SKCCombinatorialPipeline_properties():
    """Test the properties of the CombinatorialPipeline."""
    steps = [
        ("inverter", invert_objectives.InvertMinimize()),
        ("scaler", scalers.SumScaler(target="matrix")),
        ("agg", simple.WeightedSumModel()),
    ]
    pipeline = SKCCombinatorialPipeline(steps)

    assert isinstance(pipeline.steps, list)
    assert len(pipeline.steps) == 3

    assert isinstance(pipeline.named_steps, Bunch)
    assert len(pipeline.named_steps) == 3

    assert isinstance(pipeline.pipelines, list)
    assert len(pipeline.pipelines) == 1

    assert isinstance(pipeline.named_pipelines, Bunch)
    assert len(pipeline.named_pipelines) == 1
    assert len(pipeline) == 3


def test_mkcombinatorial():
    """Test the mkcombinatorial function."""
    pipeline = mkcombinatorial(
        invert_objectives.InvertMinimize(),
        [
            scalers.SumScaler(target="matrix"),
            scalers.VectorScaler(target="matrix"),
        ],
        simple.WeightedSumModel(),
    )

    assert isinstance(pipeline, SKCCombinatorialPipeline)
    assert len(pipeline.pipelines) == 2

    ranks_names = [p[0].lower() for p in pipeline.pipelines]
    assert "invertminimize_sumscaler_weightedsummodel" in ranks_names
    assert "invertminimize_vectorscaler_weightedsummodel" in ranks_names
