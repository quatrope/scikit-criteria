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
from skcriteria.combinatorial import CombinatorialPipeline
from skcriteria.preprocessing import invert_objectives, scalers
from skcriteria.preprocessing.invert_objectives import InvertMinimize


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def decision_matrix():
    dm = skc.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[1, 1, 1],
    )
    return dm


# =============================================================================
# TESTS
# =============================================================================


def test_CombinatorialPipeline_creation():

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

    pipeline = CombinatorialPipeline(steps)
    assert len(pipeline.pipelines) == 2


def test_SKCCombinatorialPipeline_evaluate(decision_matrix):

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

    pipeline = CombinatorialPipeline(steps)
    result = pipeline.evaluate(decision_matrix)

    assert len(result) == 2
    ranks_names = [r[0] for r in result.ranks]
    assert "InvertMinimize_SumScaler_WeightedSumModel" in ranks_names
    assert "InvertMinimize_VectorScaler_WeightedSumModel" in ranks_names
