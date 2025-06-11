#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.copras."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.copras import COPRAS

from skcriteria.preprocessing.scalers import SumScaler, scale_by_sum

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_COPRAS_Uysal2022assistants():
    """
    Data from:
        Uysal, Ö., & İnan, T. (2022).
        Performance Evaluation Of Research Assistants By Copras Method.
        International Journal of Social Science and Economic Research.
    """

    equal_weight = 1 / 7

    dm = skcriteria.mkdm(
        matrix=scale_by_sum(
            [
                [3.57, 4.00, 4.00, 83.75, 3, 9, 1],
                [3.07, 3.95, 4.00, 83.00, 3, 1, 3],
                [3.23, 3.54, 3.46, 66.00, 4, 0, 2],
                [3.42, 3.96, 4.00, 70.00, 5, 5, 7],
                [2.56, 3.37, 3.79, 82.00, 4, 4, 5],
            ],
            axis=0,
        ),
        objectives=[max, max, max, max, min, max, max],
        weights=[equal_weight] * 7,
        alternatives=["x1", "x2", "x3", "x4", "x5"],
        criteria=[
            "Undergraduate GPA",
            "Master GPA",
            "PhD GPA",
            "Foreign Language",
            "Lesson Completion Duration",
            "Number of Congress",
            "Number of Essays",
        ],
    )

    ranker = COPRAS()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "COPRAS",
        ["x1", "x2", "x3", "x4", "x5"],
        [1, 4, 5, 2, 3],
        extra={"score": [100.0000, 78.8501, 63.3837, 98.6506, 86.8878]},
    )

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_["score"], expected.e_["score"], atol=1e-3)
