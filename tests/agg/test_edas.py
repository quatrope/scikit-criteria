#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.edas

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.edas import EDAS
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler


def test_EDAS():
    dm = skcriteria.mkdm(
        matrix=[
            [60, 0.40, 125, 5],  # Robot 1
            [60, 0.40, 125, 6],  # Robot 2
            [68, 0.13, 75, 6],   # Robot 3
            [50, 1.00, 100, 6],  # Robot 4
            [30, 0.60, 55, 5]    # Robot 5
        ],
        objectives=[max, min, max, max],  # LC↑, R↓, VR↑, DF↑
        weights=[0.0963, 0.5579, 0.0963, 0.2495],  # wLC, wR, wVR, wDF
        criteria=["Load Capacity", "Reach", "Vibration", "Dexterity"],
        alternatives=["R1", "R2", "R3", "R4", "R5"]
    )

    expected = RankResult(
        "EDAS",
        ["R1", "R2", "R3", "R4", "R5"],
        [3, 2, 1, 5, 4],
        {
            "score": [
                0.69,
                0.69,
                0.97,
                0.007,
                0.30
            ]
        }
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)
    print(result)
    print(result.e_.score)