#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.edas"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest
# TODO: limpiar imports
import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.edas import EDAS
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler

# TODO: terminar tests
def test_EDAS():
    dm = skcriteria.mkdm(
        matrix=[
            [60, 0.40, 125, 5],  # Robot 1
            [60, 0.40, 125, 6],  # Robot 2
            [68, 0.13, 75, 6],  # Robot 3
            [50, 1.00, 100, 6],  # Robot 4
            [30, 0.60, 55, 5],  # Robot 5
        ],
        objectives=[max, min, max, max],  
        weights=[0.0963, 0.5579, 0.0963, 0.2495], 
        criteria=["Load Capacity", "Reach", "Vibration", "Dexterity"],
        alternatives=["R1", "R2", "R3", "R4", "R5"],
    )

    expected = RankResult(
        "EDAS",
        ["R1", "R2", "R3", "R4", "R5"],
        [3, 2, 1, 5, 4],
        {"score": [0.69, 0.69, 0.97, 0.007, 0.30]},
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)
    print('\n')
    print(result.e_.score)
    print(result)

def test_EDAS_Mobile_selection():
    """Test EDAS for mobile phone selection problem."""
    dm = skcriteria.mkdm(
        matrix=[
            [250, 16, 12, 5],  # Mobile-1
            [200, 16, 8, 3],   # Mobile-2
            [300, 32, 16, 4],  # Mobile-3
            [275, 32, 8, 4],   # Mobile-4
            [225, 16, 16, 2]   # Mobile-5
        ],
        objectives=[min, max, max, max],  
        weights=[0.35, 0.25, 0.25, 0.15],  
        criteria=["Price", "Storage", "Camera", "Looks"],
        alternatives=["M-1", "M-2", "M-3", "M-4", "M-5"]
    )

    # Execute EDAS
    ranker = EDAS()
    result = ranker.evaluate(dm)
    print('\n')
    print(result.e_.score)
    print(result)