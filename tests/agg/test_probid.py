#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.probid."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.probid import PROBID, SimplifiedPROBID

# =============================================================================
# TESTS
# =============================================================================


def test_SimplifiedPROBID_Wang2021Original():
    """
    Data from:
        Wang, Z., Rangaiah, G. P., & Wang, X. (2021).
        Preference ranking on the basis of ideal-average distance method for
        multi-criteria decision-making.
        Industrial & Engineering Chemistry Research, 60(30), 11216–11230.
    """
    dm = skcriteria.mkdm(
        matrix=[
            [0.1299, 0.1754, 0.3256, 0.3255, 0.0892],
            [0.1712, 0.1500, 0.2825, 0.2827, 0.2029],
            [0.1903, 0.1662, 0.3349, 0.3358, 0.0169],
            [0.2207, 0.1772, 0.3450, 0.3449, 0.0488],
            [0.2403, 0.1751, 0.3284, 0.3294, 0.0889],
            [0.2764, 0.1690, 0.2865, 0.2866, 0.1951],
            [0.3041, 0.2274, 0.2719, 0.2723, 0.3481],
            [0.3390, 0.1486, 0.2731, 0.2736, 0.3037],
            [0.3858, 0.1944, 0.3274, 0.3281, 0.2587],
            [0.4251, 0.6560, 0.2618, 0.2593, 0.5786],
            [0.4448, 0.5353, 0.2622, 0.2606, 0.5358],
        ],
        objectives=[max, min, min, min, min],
        weights=[0.1819, 0.2131, 0.1838, 0.1832, 0.2379],
    )

    ranker = SimplifiedPROBID()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "SimplifiedPROBID",
        [
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
        ],
        [5, 7, 3, 2, 1, 4, 9, 8, 6, 11, 10],
        extra={
            "score": [
                2.4246,
                2.0596,
                3.2805,
                3.3702,
                3.4375,
                2.6435,
                1.2628,
                1.8158,
                2.0885,
                0.3399,
                0.4279,
            ]
        },
    )

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_["score"], expected.e_["score"], atol=1e-2)


def test_PROBID_Wang2021Original():
    """
    Data from:
        Wang, Z., Rangaiah, G. P., & Wang, X. (2021).
        Preference ranking on the basis of ideal-average distance method for
          multi-criteria decision-making.
        Industrial & Engineering Chemistry Research, 60(30), 11216–11230.
    """
    dm = skcriteria.mkdm(
        matrix=[
            [0.1299, 0.1754, 0.3256, 0.3255, 0.0892],
            [0.1712, 0.1500, 0.2825, 0.2827, 0.2029],
            [0.1903, 0.1662, 0.3349, 0.3358, 0.0169],
            [0.2207, 0.1772, 0.3450, 0.3449, 0.0488],
            [0.2403, 0.1751, 0.3284, 0.3294, 0.0889],
            [0.2764, 0.1690, 0.2865, 0.2866, 0.1951],
            [0.3041, 0.2274, 0.2719, 0.2723, 0.3481],
            [0.3390, 0.1486, 0.2731, 0.2736, 0.3037],
            [0.3858, 0.1944, 0.3274, 0.3281, 0.2587],
            [0.4251, 0.6560, 0.2618, 0.2593, 0.5786],
            [0.4448, 0.5353, 0.2622, 0.2606, 0.5358],
        ],
        objectives=[max, min, min, min, min],
        weights=[0.1819, 0.2131, 0.1838, 0.1832, 0.2379],
    )
    ranker = PROBID()
    result = ranker.evaluate(dm)

    expected = RankResult(
        "PROBID",
        [
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
        ],
        [5, 6, 3, 2, 1, 4, 9, 8, 7, 11, 10],
        extra={
            "score": [
                0.8568,
                0.7826,
                0.9362,
                0.9369,
                0.9379,
                0.8716,
                0.5489,
                0.7231,
                0.7792,
                0.3331,
                0.3387,
            ]
        },
    )

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_["score"], expected.e_["score"], atol=1e-2)


def test_PROBID_minimize_warning():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = PROBID()
    with pytest.warns(UserWarning):
        ranker.evaluate(dm)


def test_PROBID_invalid_metric():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    with pytest.raises(ValueError):
        ranker = PROBID(metric="manhatttttan")
        ranker.evaluate(dm)


def test_SimplifiedPROBID_invalid_metric():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    with pytest.raises(ValueError):
        ranker = PROBID(metric="euuuclideann")
        ranker.evaluate(dm)


def test_SimplifiedPROBID_small_dataset_warning():
    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = SimplifiedPROBID()
    with pytest.warns(UserWarning):
        ranker.evaluate(dm)
