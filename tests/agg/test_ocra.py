#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
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
        expected.e_.performance,
    )
    assert np.allclose(
        np.round(result.e_.input_performance, 3),
        expected.e_.input_performance,
    )
    assert np.allclose(
        np.round(result.e_.output_performance, 3),
        expected.e_.output_performance,
    )
