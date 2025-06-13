#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.codas."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


import pytest


import skcriteria as skc
from skcriteria.agg import RankResult

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from skcriteria.agg.codas import CODAS
#from skcriteria.preprocessing.scalers import MinMaxScaler

def test_codas_LISCO():
    #COMPLETAR
    """
    Data From:

    """
    dm = skc.mkdm(
        matrix=[
            [45, 3600, 45, 0.9],
            [25, 3800, 60, 0.8],
            [23, 3100, 35, 0.9],
            [14, 3400, 50, 0.7],
            [15, 3300, 40, 0.8],
            [28, 3000, 30, 0.6],
        ],
        objectives=[max, min, max, max],
        alternatives=["A1", "A2", "A3", "A4", "A5", "A6"],
        criteria=["Q", "C", "LT", "LS"],
        weights=[0.2857, 0.3036, 0.2321, 0.1786],
    )


    ##transformer = MinMaxScaler(target="matrix")
    #dm = transformer.transform(dm)
    ranker = CODAS()
    result = ranker.evaluate(dm)
