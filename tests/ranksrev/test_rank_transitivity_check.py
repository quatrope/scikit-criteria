#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tests for the functionalities in the transitivity_check file."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg import RankResult
from skcriteria.agg.simple import WeightedSumModel
from skcriteria.agg.topsis import TOPSIS
from skcriteria.cmp import RanksComparator
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing.filters import FilterGE, FilterNonDominated
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler
from skcriteria.preprocessing.weighters import EntropyWeighter
from skcriteria.ranksrev.rank_transitivity_check import (
    RankTransitivityChecker,
)


# =============================================================================
# TESTS
# =============================================================================


def test_RankTransitivityChecker_creation():
    pipe = mkpipe(
        FilterGE({"ROE": 2}),
        FilterNonDominated(),
        InvertMinimize(),
        SumScaler(target="weights"),
        SumScaler(target="matrix"),
        WeightedSumModel(),
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=True)

    assert dec.allow_missing_alternatives is True
    assert dec.max_ranks == 50
    assert dec.fas_method == "auto"
    assert dec.preferred_parallel_backend is None
    assert dec.n_jobs is None

    with pytest.deprecated_call():
        assert dec.parallel_backend is None


def test_RankTransitivityChecker_simple_stock_selection():
    pipe = mkpipe(
        FilterGE({"ROE": 2}),
        FilterNonDominated(),
        InvertMinimize(),
        SumScaler(target="weights"),
        SumScaler(target="matrix"),
        WeightedSumModel(),
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=True)

    dm = skc.datasets.load_simple_stock_selection()

    result = dec.evaluate(dm)

    expected = RanksComparator(
        [
            (
                "Original",
                RankResult(
                    method="WeightedSumModel",
                    alternatives=["PE", "JN", "AA", "FX", "MM", "GN"],
                    values=np.array([3, 4, 2, 5, 5, 1], dtype=int),
                    extra={},
                ),
            ),
            (
                "Recomposition.0",
                RankResult(
                    method="Recomposition.0",
                    alternatives=["PE", "JN", "AA", "FX", "MM", "GN"],
                    values=np.array([3, 4, 2, 5, 5, 1], dtype=int),
                    extra={},
                ),
            ),
        ],
        extra={},
    )

    skc.testing.assert_rcmp_equals(expected, result, skip_extra=True)


def test_RankTransitivityChecker_allow_missing_alternative_false():
    pipe = mkpipe(
        FilterGE({"ROE": 2}),
        FilterNonDominated(),
        InvertMinimize(),
        SumScaler(target="weights"),
        SumScaler(target="matrix"),
        WeightedSumModel(),
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=False)

    dm = skc.datasets.load_simple_stock_selection()

    with pytest.raises(ValueError, match="Missing alternative/s"):
        dec.evaluate(dm)


def test_RankTransitivityChecker_vanheerden():
    pipe = mkpipe(
        InvertMinimize(),
        EntropyWeighter(),
        SumScaler(target="matrix"),
        TOPSIS(),
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=True)

    dm = skc.datasets.load_van2021evaluation()

    result = dec.evaluate(dm)

    # Basic assertions to verify the result structure
    assert isinstance(result, RanksComparator)
    assert len(result) >= 1  # At least one rank (Original)

    # Check that all ranks have the same alternatives
    first_alternatives = result.ranks[0][1].alternatives
    for name, rank in result.ranks:
        np.testing.assert_array_equal(rank.alternatives, first_alternatives)
        assert rank.method in ("TOPSIS", name)


def test_RankTransitivityChecker_bad_pipe():
    bad_pipe = "Suffering and pain"

    with pytest.raises(
        TypeError, match="'dmaker' must implement 'evaluate\\(\\)' method"
    ):
        RankTransitivityChecker(bad_pipe)


def test_RankTransitivityChecker_parallel_backend():
    pipe = mkpipe(
        InvertMinimize(),
        EntropyWeighter(),
        SumScaler(target="matrix"),
        TOPSIS(),
    )
    with pytest.deprecated_call():
        RankTransitivityChecker(pipe, parallel_backend="pika")

    with pytest.raises(ValueError):
        RankTransitivityChecker(
            pipe, parallel_backend="pika", preferred_parallel_backend="pika"
        )


def test_RankTransitivityChecker_max_rank_lt_1():
    pipe = mkpipe(
        InvertMinimize(),
        EntropyWeighter(),
        SumScaler(target="matrix"),
        TOPSIS(),
    )

    with pytest.raises(ValueError):
        RankTransitivityChecker(pipe, max_ranks=0)


def test_RankTransitivityChecker_repr():
    pipe = mkpipe(
        InvertMinimize(),
        TOPSIS(),
    )

    expected = (
        "<RankTransitivityChecker "
        "<SKCPipeline [steps=[('invertminimize', <InvertMinimize []>), "
        "('topsis', <TOPSIS [metric='euclidean']>)]]>, "
        "fas_method=auto, max_ranks=50>"
    )

    dec = RankTransitivityChecker(pipe)

    assert repr(dec) == expected
