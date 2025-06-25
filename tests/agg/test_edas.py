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
import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.edas import EDAS


# TODO: terminar tests

def test_EDAS_robot_1():
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

    ranker = EDAS()
    result = ranker.evaluate(dm)
    print('\n')
    print(result)
    print(result.e_.score)


def test_EDAS_robot_example_2():
    weights = [0.1574, 0.1825, 0.2385, 0.2172, 0.2044]

    dm = skcriteria.mkdm(
        matrix=[
            [60.0, 0.40, 2540, 500, 990],   # R1
            [6.35, 0.15, 1016, 3000, 1041], # R2
            [6.80, 0.10, 1727.2, 1500, 1676], # R3
            [10.0, 0.20, 1000, 2000, 965],  # R4
            [2.50, 0.10, 560, 500, 915],    # R5
            [4.50, 0.08, 1016, 350, 508],   # R6
            [3.00, 0.10, 177, 1000, 920],   # R7
        ],
        objectives=[max, min, max, max, max],
        weights=weights,
        criteria=["Load Capacity", "Repeatability", "Max Tip Speed", "Memory", "Reach"],
        alternatives=["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    print(result)
    print(result.e_.score)



def test_EDAS_robot_example3():
    # Note: Qualitative criteria (MI, PF, SC) are converted to numerical scores as per paper
    dm = skcriteria.mkdm(
        matrix=[
            [70, 45, 0.16, 0.6818, 0.8636, 1.0],   # Robot 1 (AA, H, VH)
            [68, 45, 0.17, 0.6818, 1.0, 0.6818],    # Robot 2 (AA, VH, AA)
            [73, 50, 0.12, 0.8636, 0.8636, 0.6818], # Robot 3 (H, H, AA)
        ],
        objectives=[min, max, min, max, max, max],  # PC(min), LC(max), R(min), MI(max), PF(max), SC(max)
        weights=[0.1830, 0.1009, 0.3833, 0.0555, 0.1027, 0.1746],  # Weights from paper
        criteria=[
            "Purchase Cost ($1000)", 
            "Load Capacity (kg)", 
            "Repeatability Error (mm)", 
            "Man-Machine Interface", 
            "Programming Flexibility", 
            "Service Contract"
        ],
        alternatives=["Robot 1", "Robot 2", "Robot 3"],
    )

    # Expected ranking from paper (Table 7): 2-3-1 (Cincinnati > Cybotech > ASEA)
    ranker = EDAS()
    result = ranker.evaluate(dm)
    
    print('\nEDAS Ranking Results for Example 3:')
    print('Ranking:', result)
    print('Scores:', result.e_.score)


def test_EDAS_mobile_selection():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 16, 12, 5],
            [200, 16, 8, 3],
            [300, 32, 16, 4],
            [275, 32, 8, 4],
            [225, 16, 16, 2]
        ],
        objectives=[min, max, max, max],
        weights=[0.35, 0.25, 0.25, 0.15]
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)
    print(result)


def test_EDAS_CODAS():
    dm = skcriteria.mkdm(
        matrix=[
            [256, 8, 41, 1.6, 1.77, 7347.16],  # A1
            [256, 8, 32, 1.0, 1.8, 6919.99],    # A2
            [256, 8, 53, 1.6, 1.9, 8400],       # A3
            [256, 8, 41, 1.0, 1.75, 6808.9],    # A4
            [512, 8, 35, 1.6, 1.7, 8479.99],    # A5
            [256, 4, 35, 1.6, 1.7, 7499.99]    # A6
        ],
        objectives=[max, max, max, max, min, min],  # C1-C6 objectives
        weights=[0.405, 0.221, 0.134, 0.199, 0.007, 0.034]  # Weights from imagen.png
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    print(result)

def test_EDAS_1():
    dm = skcriteria.mkdm(
        matrix=[
        # C1   C2     C3    C4    C5    C6      C7
        [23,   264,   2.37, 0.05, 167,  8900,   8.71],  # A1
        [20,   220,   2.20, 0.04, 171,  9100,   8.23],  # A2
        [17,   231,   1.98, 0.15, 192,  10800,  9.91],  # A3
        [12,   210,   1.73, 0.20, 195,  12300, 10.21],  # A4
        [15,   243,   2.00, 0.14, 187,  12600,  9.34],  # A5
        [14,   222,   1.89, 0.13, 180,  13200,  9.22],  # A6
        [21,   262,   2.43, 0.06, 160,  10300,  8.93],  # A7
        [20,   256,   2.60, 0.07, 163,  11400,  8.44],  # A8
        [19,   266,   2.10, 0.06, 157,  11200,  9.04],  # A9
        [8,    218,   1.94, 0.11, 190,  13400, 10.11],  # A10
    ],
    objectives=[max, max, max, min, min, min, min],  # C1-C3: benefit; C4-C7: cost
    weights=[0.250, 0.214, 0.179, 0.143, 0.107, 0.071, 0.036],  # Set 1 weights from Table 6
    criteria=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    alternatives=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    )

    ranker = EDAS()
    result = ranker.evaluate(dm)

    print(result)