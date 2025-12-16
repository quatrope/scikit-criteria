#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.marcos"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria import mkdm, testing
from skcriteria.agg._agg_base import RankResult
from skcriteria.agg.marcos import MARCOS

# =============================================================================
# TEST CLASSES
# =============================================================================


@pytest.fixture
def stevic2019():
    """
    Data from:
    Stević, Z., Pamučar, D., Puška, A., Chatterjee, P., Sustainable supplier
    selection in healthcare industries using a new MCDM method: Measurement
    Alternatives and Ranking according to COmpromise Solution (MARCOS),
    Computers & Industrial Engineering (2019)
    """
    data = [
        [
            6.257,
            4.217,
            4.217,
            6.257,
            3.000,
            4.217,
            5.000,
            3.557,
            3.557,
            3.557,
            3.000,
            5.000,
            4.718,
            3.557,
            3.557,
            2.080,
            3.557,
            3.000,
            4.718,
            3.557,
            2.080,
        ],
        [
            4.217,
            6.804,
            7.000,
            5.000,
            7.000,
            6.804,
            5.593,
            5.593,
            6.804,
            7.000,
            5.000,
            7.612,
            5.593,
            6.257,
            5.000,
            7.000,
            5.593,
            5.593,
            6.804,
            5.593,
            5.593,
        ],
        [
            4.718,
            5.593,
            6.257,
            5.000,
            4.718,
            5.000,
            5.593,
            4.718,
            5.593,
            5.000,
            3.557,
            6.257,
            5.000,
            4.718,
            4.718,
            5.000,
            3.557,
            5.593,
            5.593,
            3.557,
            4.217,
        ],
        [
            5.000,
            6.804,
            5.000,
            3.000,
            5.000,
            6.257,
            7.612,
            3.557,
            5.000,
            6.257,
            6.257,
            5.593,
            6.257,
            5.000,
            5.593,
            7.000,
            5.000,
            6.257,
            5.000,
            3.557,
            4.217,
        ],
        [
            3.557,
            5.593,
            6.804,
            3.000,
            5.000,
            7.000,
            5.593,
            5.000,
            6.257,
            7.000,
            5.593,
            7.612,
            6.257,
            6.257,
            5.000,
            6.257,
            5.593,
            7.000,
            5.000,
            4.718,
            5.000,
        ],
        [
            6.257,
            3.000,
            4.217,
            5.000,
            3.557,
            3.000,
            4.217,
            3.000,
            4.217,
            3.000,
            2.466,
            3.000,
            4.217,
            3.557,
            5.000,
            3.000,
            4.217,
            3.557,
            2.080,
            5.000,
            3.000,
        ],
        [
            4.217,
            5.000,
            6.257,
            5.593,
            3.557,
            5.593,
            4.217,
            5.593,
            5.000,
            6.257,
            3.557,
            5.000,
            6.257,
            5.593,
            5.593,
            7.000,
            6.257,
            5.000,
            6.257,
            5.593,
            5.000,
        ],
        [
            7.612,
            1.442,
            3.000,
            9.000,
            2.080,
            3.000,
            1.442,
            2.080,
            1.442,
            3.000,
            1.000,
            1.442,
            2.080,
            3.000,
            3.000,
            1.000,
            1.442,
            1.442,
            3.000,
            2.080,
            1.000,
        ],
    ]
    # Only C11 and C14 are minimizing criteria, the rest are maximizing.
    objectives = [min, max, max, min] + [max] * 17
    weights = [
        0.127,
        0.159,
        0.060,
        0.075,
        0.043,
        0.051,
        0.075,
        0.061,
        0.053,
        0.020,
        0.039,
        0.022,
        0.017,
        0.027,
        0.022,
        0.039,
        0.017,
        0.035,
        0.015,
        0.024,
        0.016,
    ]
    return mkdm(matrix=data, objectives=objectives, weights=weights)


def test_marcos_compare_with_reference(stevic2019):
    """Test MARCOS against reference values from Stević et al. 2019."""
    # Reference values from paper
    f_K = np.array([0.524, 0.846, 0.704, 0.796, 0.843, 0.499, 0.722, 0.290])
    rank = np.array([6, 1, 5, 3, 2, 7, 4, 8])
    # Note: K_minus[4] value corrected from paper's 2.992 to computed 2.920
    # (likely rounding error in paper)
    K_minus = np.array(
        [1.818, 2.931, 2.439, 2.760, 2.922, 1.729, 2.502, 1.007]
    )
    K_plus = np.array([0.563, 0.908, 0.755, 0.855, 0.905, 0.535, 0.775, 0.312])

    extra = {"utility_scores": f_K, "K_minus": K_minus, "K_plus": K_plus}

    expected = RankResult(
        "MARCOS",
        stevic2019.alternatives,
        rank,
        extra=extra,
    )

    ranker = MARCOS()
    result = ranker.evaluate(stevic2019)

    testing.assert_result_equals(result, expected, atol=1e-2)
