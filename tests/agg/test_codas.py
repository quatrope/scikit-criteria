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
from skcriteria.preprocessing.scalers import MinMaxScaler

def test_codas_negative_coord_fail():
    dm = skc.mkdm(
        matrix=[[1, 2, 3], [4, -1, 6]],
        objectives=[max, max, max],
    )

    ranker = CODAS()

    with pytest.raises(ValueError):
        ranker.evaluate(dm)



def test_codas_LISCO():
    
    """
    Data From:

    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3177276

    The model is applied with a real-world case study for ranking the suppliers
    in the Libyan Iron and Steel Company (LISCO).

    The import of raw material is a very important step in the steelmaking
    process. The quality and cost of the final products are intimately connected
    to this initial step.

    Four different criteria considered in this supplier selection problem are:
    - Quality (in points)
    - Direct Cost (in $)
    - Lead time (in days)
    - Logistics services (in points)

    All these criteria are defined as benefit criteria, except the cost, which is
    defined as a cost criterion.

    This problem consists of six suppliers.

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

    expected = RankResult(
        "CODAS", ["A1", "A2", "A3", "A4", "A5", "A6"], [1, 2, 3, 5, 6, 4], 
        {"score":  [1.3914,  0.3411,  -0.2170, -0.5381, -0.7292, -0.2481]}
    )


    ##transformer = MinMaxScaler(target="matrix")
    #dm = transformer.transform(dm)
    ranker = CODAS()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method

    assert np.all(result.e_.score.round(4) == expected.e_.score)

def test_codas_chakraborty_zavadaskas2014():
    #dos minutos te costaba chequear que los datos esten bien zavadaskas....
    """ 
    Data From:
   
    https://www.researchgate.net/publication/308697546_A_new_combinative_distance-based_assessment_CODAS_method_for_multi-criteria_decision-making

    This example is adapted from Chakraborty and Zavadskas (2014), which is
    related to the selection of the most appropriate industrial robot.

    Five different criteria considered in this robot selection problem are:
    - Load capacity (in kg)
    - Maximum tip speed (in mm/s)
    - Repeatability (in mm)
    - Memory capacity (in points or steps)
    - Manipulator reach (in mm)

    Among these criteria, the load capacity, maximum tip speed, memory capacity,
    and manipulator reach are defined as benefit criteria, while repeatability
    is defined as a cost criterion.

    This problem consists of seven alternatives.
    """
    dm = skc.mkdm(
        matrix=[
            [60, 0.4, 2540, 500, 990],
            [6.35, 0.15, 1016, 3000, 1041],
            [6.8, 0.10, 1727.2, 1500, 1676],
            [10, 0.2, 1000, 2000, 965],
            [2.5, 0.1, 560, 500, 915],
            [4.5, 0.08, 1016, 350, 508],
            [3, 0.1, 1778, 1000, 920],

        ],
        objectives=[max, min, max, max, max],
        alternatives=["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        criteria=["LC", "MtS", "Rep", "MemC", "ManR"],
        weights=[0.036, 0.192, 0.326, 0.326, 0.120],
    )

    expected = RankResult(
        "CODAS", ["A1", "A2", "A3", "A4", "A5", "A6", "A7"], [3, 1, 2, 5, 7, 6, 4], 
        {"score":  [0.5122,  1.4633,  1.0715, -0.2125, -1.8515, -1.1717, 0.1887]}
    )
   
    ranker = CODAS()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.score.round(4) == expected.e_.score)



def test_codas_zavadaskas_turskis2010():
    #dos minutos te costaba chequear que los datos esten bien zavadaskas....
    """ 
    Data From:
    
    https://www.researchgate.net/publication/308697546_A_new_combinative_distance-based_assessment_CODAS_method_for_multi-criteria_decision-making
    
    This example is adapted from Zavadskas and Turskis (2010) and considers the
    evaluation of microclimate in an office environment.

    Six criteria determined for this evaluation process are:
    - Amount of air per head (in m³/h)
    - Relative air humidity (in percent)
    - Air temperature (in °C)
    - Illumination during work hours (in lx)
    - Rate of air flow (in m/s)
    - Dew point (in °C)

    All of these criteria are defined as benefit criteria, except the rate of air
    flow and the dew point, which are considered cost criteria.

    Fourteen alternatives should be evaluated according to these criteria.
    """
    dm = skc.mkdm(
        matrix=[
        [7.6, 46, 18, 390, 0.1, 11],
        [5.5, 32, 21, 360, 0.05, 11],
        [5.3, 32, 21, 290, 0.05, 11],
        [5.7, 37, 19, 270, 0.05, 9],
        [4.2, 38, 19, 240, 0.1, 8],
        [4.4, 38, 19, 260, 0.1, 8],
        [3.9, 42, 16, 270, 0.1, 5],
        [7.9, 44, 20, 400, 0.05, 6],
        [8.1, 44, 20, 380, 0.05, 6],
        [4.5, 46, 18, 320, 0.1, 7],
        [5.7, 48, 20, 320, 0.05, 11],
        [5.2, 48, 20, 310, 0.05, 11],
        [7.1, 49, 19, 280, 0.1, 12],
        [6.9, 50, 16, 250, 0.05, 10],
        ],
        objectives=[max, max, max, max, min, min],
        alternatives=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"],
        criteria=["A/P", "RH", "AT", "ILw", "AF", "DP"],
        weights=[0.21, 0.16, 0.26, 0.17, 0.12, 0.08],
    )
    
    expected = RankResult(
        "CODAS", ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"], 
        [3, 6, 9, 10, 14, 13, 12, 1, 2, 11, 4, 7, 8, 5], 
        {"score":  
        [0.768, 0.363, -0.105, -0.329, -2.384, -2.207, -2.043, 2.929, 2.890, -1.282, 0.568, 0.313, 0.157, 0.364]}
    )
   
    ranker = CODAS()
    result = ranker.evaluate(dm)    

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.score.round(3) == expected.e_.score)