#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.ram"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.ram import RAM
from skcriteria.preprocessing.scalers import SumScaler

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_RootAssessmentMethod():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6]],
        objectives=[max, min, max],
    )

    expected = RankResult(
        "RAM",
        ["A0", "A1"],
        [1, 2],
        {
            "sum_benefit": np.array([4.0, 10.0]),
            "sum_cost": np.array([2.0, 5.0]),
            "score": np.array([1.56508458, 1.42616164]),
        },
    )
    ranker = RAM()
    result = ranker.evaluate(dm)
    assert expected.aequals(result)


def test_RootAssessmentMethod_sotoudehanvari2023138695():
    """
    Data from:
        Sotoudeh-Anvari, A. (2023). Root Assessment Method (RAM):
        A novel multi-criteria decision making method and its applications
        in sustainability challenges.
        Journal of Cleaner Production, 423, 138695.
        Page 7, Tables 2 to 5.
        https://www.sciencedirect.com/science/article/abs/pii/S0959652623028536

    """
    dm = skcriteria.mkdm(
        matrix=[
            [0.068, 0.066, 0.150, 0.098, 0.156, 0.114, 0.098],
            [0.078, 0.076, 0.108, 0.136, 0.082, 0.171, 0.105],
            [0.157, 0.114, 0.128, 0.083, 0.108, 0.113, 0.131],
            [0.106, 0.139, 0.058, 0.074, 0.132, 0.084, 0.120],
            [0.103, 0.187, 0.125, 0.176, 0.074, 0.064, 0.057],
            [0.105, 0.083, 0.150, 0.051, 0.134, 0.094, 0.113],
            [0.137, 0.127, 0.056, 0.133, 0.122, 0.119, 0.114],
            [0.100, 0.082, 0.086, 0.060, 0.062, 0.109, 0.093],
            [0.053, 0.052, 0.043, 0.100, 0.050, 0.078, 0.063],
            [0.094, 0.074, 0.097, 0.087, 0.080, 0.054, 0.106],
        ],
        objectives=[max, min, min, max, max, max, max],
        weights=[0.132, 0.135, 0.138, 0.162, 0.09, 0.223, 0.12],
    )

    expected = RankResult(
        "RAM",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
        [4, 2, 3, 5, 10, 7, 1, 6, 8, 9],
        {
            "sum_benefit": np.array(
                [
                    0.07609,
                    0.090475,
                    0.08481,
                    0.071,
                    0.06992,
                    0.0687,
                    0.09085,
                    0.06397,
                    0.05267,
                    0.05848,
                ]
            ),
            "sum_cost": np.array(
                [
                    0.029589,
                    0.025149,
                    0.03303,
                    0.02676,
                    0.04247,
                    0.03188,
                    0.02486,
                    0.02292,
                    0.01294,
                    0.02336,
                ]
            ),
            "score": np.array(
                [
                    1.433215,
                    1.439243,
                    1.435296,
                    1.432197,
                    1.42788,
                    1.43012,
                    1.439444,
                    1.430766,
                    1.429406,
                    1.428773,
                ]
            ),
        },
    )

    transformer = SumScaler(target="both")
    dm = transformer.transform(dm)

    ranker = RAM()
    result = ranker.evaluate(dm)

    assert expected.aequals(result, rtol=1e-5, atol=1e-5)
