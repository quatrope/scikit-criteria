#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.ocra."""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.ocra import OCRA


# =============================================================================
# OCRA
# =============================================================================


def test_OCRA_hotels():
    """
    Data From:
        Işık, A. T., & Adalı, E. A. A new integrated decision making approach
        based on SWARA and OCRA methods for the hotel selection problem. -
        International Journal of Advanced Operations Management (2016).
    """
    dm = skcriteria.mkdm(
        matrix=[
            [7.7, 256, 7.2, 7.3, 7.3],
            [8.1, 250, 7.9, 7.8, 7.7],
            [8.7, 352, 8.6, 7.9, 8.0],
            [8.1, 262, 7.0, 8.1, 7.2],
            [6.5, 271, 6.3, 6.4, 6.1],
            [6.8, 228, 7.1, 7.2, 6.5],
        ],
        objectives=[max, min, max, max, max],
        weights=[0.239, 0.225, 0.197, 0.186, 0.153],
        alternatives=["A1", "A2", "A3", "A4", "A5", "A6"],
        criteria=["C1", "C2", "C3", "C4", "C5"],
    )

    expected = RankResult(
        "OCRA",
        ["A1", "A2", "A3", "A4", "A5", "A6"],
        [4, 1, 3, 2, 6, 5],
        {
            "performance": [0.143, 0.210, 0.164, 0.167, 0.000, 0.112],
            "input_performance": [0.095, 0.101, 0.000, 0.089, 0.080, 0.122],
            "output_performance": [0.129, 0.190, 0.244, 0.158, 0.000, 0.069],
        },
    )

    ranker = OCRA()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(
        np.round(result.e_.performance, 3),
        expected.e_.performance
    )
    assert np.allclose(
        np.round(result.e_.input_performance, 3),
        expected.e_.input_performance
    )
    assert np.allclose(
        np.round(result.e_.output_performance, 3),
        expected.e_.output_performance
    )


def test_OCRA_NCMP():
    """
    Data From:
        Madić, M., Petković, D., Radovanović, M. Selection of non-conventional
        machining processes using the OCRA method. -
        Serbian Journal of Management (2015).
        Final performance values were adjusted to have a zero minimum!
    """
    dm = skcriteria.mkdm(
        matrix=[
            [1, 10, 500, 2, 4, 2, 3, 1, 4, 1],
            [2.5, 0.22, 0.8, 1, 4, 2, 2, 3, 3, 1],
            [2.5, 0.24, 0.5, 1, 4, 2, 2, 3, 3, 1],
            [3, 100, 400, 5, 2, 3, 1, 3, 5, 4],
            [3, 0.4, 15, 3, 3, 2, 1, 3, 5, 4],
            [3.5, 2.7, 800, 3, 4, 4, 4, 3, 4, 5],
            [3.5, 2.5, 600, 3, 4, 4, 4, 3, 4, 5],
            [2.5, 0.2, 1.6, 4, 5, 2, 1, 3, 4, 1],
            [2, 1.4, 0.1, 3, 5, 2, 1, 1, 4, 1],
        ],
        objectives=[min, min, max, min, max, min, min, max, max, max],
        weights=[0.0783, 0.0611, 0.1535, 0.1073, 0.0383, 0.0271, 0.0195,
            0.0146, 0.2766, 0.2237],
        alternatives=["USM", "WJM", "AJM", "ECM", "CHM", "EDM", "WEDM", "EBM",
            "LBM"],
        criteria=["TSF", "PR", "MRR", "C", "E", "TF", "TC", "S", "M", "F"],
    )

    expected = RankResult(
        "OCRA",
        ["USM", "WJM", "AJM", "ECM", "CHM", "EDM", "WEDM", "EBM", "LBM"],
        [3, 7, 8, 4, 5, 1, 2, 6, 9],
        {
            "performance": [767.0009, 1.2132, 0.7516, 608.8521, 23.6024,
                1228.4768, 921.4878, 2.2512, 0.0000],
            "input_performance": [5.4031, 5.9503, 5.9492, 0.0000, 5.7062,
                5.4549, 5.4659, 5.6490, 5.7295],
            "output_performance": [767.3274, 0.9924, 0.5319, 614.5816,
                23.6257, 1228.7514, 921.7514, 2.3317, 0.0000],
        },
    )

    ranker = OCRA()
    result = ranker.evaluate(dm)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(
        np.round(result.e_.performance, 4),
        expected.e_.performance
    )
    assert np.allclose(
        np.round(result.e_.input_performance, 4),
        expected.e_.input_performance
    )
    assert np.allclose(
        np.round(result.e_.output_performance, 4),
        expected.e_.output_performance
    )