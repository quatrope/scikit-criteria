#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Test suite for the SPOTIS method.

Many tests are based on the reference examples from the method paper:

Dezert, J., Tchamova, A., Han, D., & Tacnet, J. M. (2020, July).
The SPOTIS rank reversal free method for multi-criteria
decision-making support. In 2020 IEEE 23rd International Conference
on Information Fusion (FUSION) (pp. 1-8). IEEE.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.spotis import SPOTIS


def test_reference_example_a():
    dm = skcriteria.mkdm(
        matrix=[
            [10.5, -3.1, 1.7],
            [-4.7, 0, 3.4],
            [8.1, 0.3, 1.3],
            [3.2, 7.3, -5.3],
        ],
        objectives=[max, min, max],
        weights=[0.2, 0.3, 0.5],
        criteria=["C1", "C2", "C3"],
    )
    bounds = np.array([[-5, 12], [-6, 10], [-8, 5]])
    isp = np.array([12, -6, 5])

    result = SPOTIS().evaluate(dm, bounds=bounds, isp=isp)
    expected = RankResult(
        "SPOTIS",
        ["A0", "A1", "A2", "A3"],
        [1, 3, 2, 4],
        {
            "score": [0.1989, 0.3707, 0.3063, 0.7491],
            "isp": isp,
            "bounds": bounds,
        },
    )

    __assert_rank_result(result, expected, atol=1e-3)


def test_reference_example_b():
    dm = skcriteria.mkdm(
        matrix=[
            [15000, 4.3, 99, 42, 737],
            [15290, 5.0, 116, 42, 892],
            [15350, 5.0, 114, 45, 952],
            [15490, 5.3, 123, 45, 1120],
        ],
        objectives=[min, min, min, max, max],
        weights=[0.2941, 0.2353, 0.2353, 0.0588, 0.1765],
        criteria=["C1", "C2", "C3", "C4", "C5"],
    )
    bounds = np.array(
        [[14000, 16000], [3, 8], [80, 140], [35, 60], [650, 1300]]
    )
    isp = np.array([14000, 3, 80, 60, 1300])

    result = SPOTIS().evaluate(dm, bounds=bounds, isp=isp)
    expected = RankResult(
        "SPOTIS",
        ["A0", "A1", "A2", "A3"],
        [1, 3, 2, 4],
        {
            "score": [0.4779, 0.5781, 0.5558, 0.5801],
            "isp": isp,
            "bounds": bounds,
        },
    )

    __assert_rank_result(result, expected)


def test_bounds_from_matrix():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [10, -1, 4]],
        objectives=[max, min, max],
        weights=[1, 1, 1],
        criteria=["C1", "C2", "C3"],
    )

    # Test that the bounds are calculated from the matrix
    result = SPOTIS().evaluate(dm)
    assert np.all(result.e_.bounds == np.array([[1, 10], [-1, 5], [3, 6]]))


def test_isp_from_bounds():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [10, -1, 4]],
        objectives=[max, min, max],
        weights=[1, 1, 1],
        criteria=["C1", "C2", "C3"],
    )

    # Test that the ISP is calculated from the bounds given by the matrix
    result = SPOTIS().evaluate(dm)
    assert np.all(result.e_.isp == np.array([10, -1, 6]))

    # Test that the ISP is calculated from the bounds provided
    bounds = np.array([[0, 15], [-5, 11], [3, 6]])
    result = SPOTIS().evaluate(dm, bounds=bounds)
    assert np.all(result.e_.isp == np.array([15, -5, 6]))


def test_bounds_validation():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[max, min, max],
        weights=[1, 1, 1],
        criteria=["C1", "C2", "C3"],
    )

    # Note that the bounds are invalid because the shape is wrong.
    bounds = np.array(
        [
            [1, 6],
        ]
    )
    with pytest.raises(ValueError, match=r"Invalid shape for bounds.*"):
        SPOTIS().evaluate(dm, bounds=bounds)

    # Note that the bounds are invalid because there is an
    # alternative with a value out of the bounds.
    bounds = np.array([[1, 6], [2, 8], [3, 9]])

    with pytest.raises(
        ValueError,
        match="The matrix values must be within the provided bounds.",
    ):
        SPOTIS().evaluate(dm, bounds=bounds)


def test_isp_validation():
    dm = skcriteria.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[max, min, max],
        weights=[1, 1, 1],
        criteria=["C1", "C2", "C3"],
    )

    # Note that the ISP is invalid because it has an invalid shape.
    bounds = np.array([[1, 7], [2, 8], [3, 9]])
    isp = np.array([0, 1])

    with pytest.raises(
        ValueError, match=r"Invalid shape for Ideal Solution Point \(ISP\).*"
    ):
        SPOTIS().evaluate(dm, bounds=bounds, isp=isp)

    # Note that the ISP is invalid because it is outside the
    # bounds of the criterias.
    bounds = np.array([[1, 7], [2, 8], [3, 9]])
    isp = np.array([0, 2, 3])

    with pytest.raises(
        ValueError,
        match="The isp values must be within the provided bounds.",
    ):
        SPOTIS().evaluate(dm, bounds=bounds, isp=isp)


def __assert_rank_result(result, expected, atol=1e-4):
    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.allclose(result.e_.score, expected.e_.score, atol=atol)
    assert np.all(result.e_.bounds == expected.e_.bounds)
    assert np.all(result.e_.isp == expected.e_.isp)
