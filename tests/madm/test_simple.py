#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.add_value_to_zero

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.data import RankResult
from skcriteria.madm import WeightedSumModel
from skcriteria.preprocessing import MinimizeToMaximize, SumScaler

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_SAM():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, max, max],
    )

    expected = RankResult(
        "WeightedSumModel", ["A0", "A1"], [2, 1], {"score": [4.0, 11.0]}
    )

    ranker = WeightedSumModel()

    result = ranker.rank(dm)

    assert result.equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.score == expected.e_.score)


def test_SAM_minimize_fail():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = WeightedSumModel()

    with pytest.raises(ValueError):
        ranker.rank(dm)


def test_SAM_kracka2010ranking():
    """
    Data from:
        KRACKA, M; BRAUERS, W. K. M.; ZAVADSKAS, E. K. Ranking
        Heating Losses in a Building by Applying the MULTIMOORA. -
        ISSN 1392 - 2785 Inzinerine Ekonomika-Engineering Economics, 2010,
        21(4), 352-359.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [33.95, 23.78, 11.45, 39.97, 29.44, 167.10, 3.852],
            [38.9, 4.17, 6.32, 0.01, 4.29, 132.52, 25.184],
            [37.59, 9.36, 8.23, 4.35, 10.22, 136.71, 10.845],
            [30.44, 37.59, 13.91, 74.08, 45.10, 198.34, 2.186],
            [36.21, 14.79, 9.17, 17.77, 17.06, 148.3, 6.610],
            [37.8, 8.55, 7.97, 2.35, 9.25, 134.83, 11.935],
        ],
        objectives=[min, min, min, min, max, min, max],
        weights=[20, 20, 20, 20, 20, 20, 20],
    )

    transformers = [
        MinimizeToMaximize(),
        SumScaler(target="both"),
    ]
    for t in transformers:
        dm = t.transform(dm)

    expected = RankResult(
        "WeightedSumModel",
        ["A0", "A1", "A2", "A3", "A4", "A5"],
        [6, 1, 3, 4, 5, 2],
        {
            "score": [
                0.12040426,
                0.3458235,
                0.13838192,
                0.12841246,
                0.12346084,
                0.14351701,
            ]
        },
    )

    ranker = WeightedSumModel()
    result = ranker.rank(dm)

    assert result.equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score)
