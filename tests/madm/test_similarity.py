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
from skcriteria.madm import TOPSIS
from skcriteria.preprocessing import VectorScaler

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_TOPSIS():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, max, max],
    )

    expected = RankResult(
        "TOPSIS",
        ["A0", "A1"],
        [2, 1],
        {
            "ideal": [1, 5, 6],
            "anti_ideal": [0, 0, 3],
            "similarity": [0.14639248, 0.85360752],
        },
    )

    ranker = TOPSIS()
    result = ranker.rank(dm)

    assert result.equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.ideal == expected.e_.ideal)
    assert np.allclose(result.e_.anti_ideal, expected.e_.anti_ideal)
    assert np.allclose(result.e_.similarity, expected.e_.similarity)


def test_TOPSIS_minimize_warning():

    dm = skcriteria.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[max, min, max],
    )

    ranker = TOPSIS()

    with pytest.warns(UserWarning):
        ranker.rank(dm)


def test_TOPSIS_tzeng2011multiple():
    """
    Data from:
        Tzeng, G. H., & Huang, J. J. (2011).
        Multiple attribute decision making: methods and applications.
        CRC press.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [5, 8, 4],
            [7, 6, 8],
            [8, 8, 6],
            [7, 4, 6],
        ],
        objectives=[max, max, max],
        weights=[0.3, 0.4, 0.3],
    )

    transformer = VectorScaler(target="matrix")
    dm = transformer.transform(dm)

    expected = RankResult(
        "TOPSIS",
        ["A0", "A1", "A2", "A3"],
        [3, 2, 1, 4],
        {"similarity": [0.5037, 0.6581, 0.7482, 0.3340]},
    )

    ranker = TOPSIS()
    result = ranker.rank(dm)

    assert result.equals(expected)
    assert result.method == expected.method
    assert np.allclose(
        result.e_.similarity, expected.e_.similarity, atol=1.0e-4
    )
