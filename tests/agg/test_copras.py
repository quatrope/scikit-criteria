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
from skcriteria.preprocessing.scalers import scale_by_sum

# =============================================================================
# TESTS
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


def test_COPRAS_NoMinimizingCriteriaExcption():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, max, max],
        weights=[1, 2, 3, 4],
    )

    ranker = COPRAS()
    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_COPRAS_NegativeValuesException():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, -40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, min, max, max],
        weights=[1, 2, 3, 4],
    )

    ranker = COPRAS()
    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_COPRAS_All0sInMinimizingCriteriaException():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 0, 40, 0],
            [350, 340, 15, 600],
        ],
        objectives=[max, min, max, min],
        weights=[1, 2, 3, 4],
    )

    ranker = COPRAS()
    with pytest.raises(ValueError):
        ranker.evaluate(dm)


def test_COPRAS_Więckowski2022CriteriaMethodsComparison():
    """
    Data from:
        Więckowski, J., & Szyjewski, Z. (2022).
        Practical Study of Selected Multi-Criteria Methods Comparison.
        Procedia Computer Science, 207, 4565–4573.
        https://doi.org/10.1016/j.procs.2022.09.520
    """

    dm = skcriteria.mkdm(
        matrix=scale_by_sum(
            [
                [3.5, 6.0, 1256.0, 4.0, 16.0, 3.0, 17.3, 8.0, 2.82, 4100.0],
                [3.1, 4.0, 1000.0, 2.0, 8.0, 1.0, 15.6, 5.0, 3.08, 3800.0],
                [3.6, 6.0, 2000.0, 4.0, 16.0, 3.0, 17.3, 5.0, 2.90, 4000.0],
                [3.0, 4.0, 1000.0, 2.0, 8.0, 2.0, 17.3, 5.0, 2.60, 3500.0],
                [3.3, 6.0, 1008.0, 4.0, 12.0, 3.0, 15.6, 8.0, 2.30, 3800.0],
                [3.6, 6.0, 1000.0, 2.0, 16.0, 3.0, 15.6, 5.0, 2.80, 4000.0],
                [3.5, 6.0, 1256.0, 2.0, 16.0, 1.0, 15.6, 6.0, 2.90, 4000.0],
            ],
            axis=0,
        ),
        objectives=[max, max, max, max, max, max, max, max, min, min],
        weights=[
            0.297,
            0.025,
            0.035,
            0.076,
            0.154,
            0.053,
            0.104,
            0.017,
            0.025,
            0.214,
        ],
    )

    ranker = COPRAS()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "COPRAS",
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
        [2, 7, 1, 6, 3, 4, 5],
        extra={},
    )

    assert result.values_equals(expected)
    assert result.method == expected.method
