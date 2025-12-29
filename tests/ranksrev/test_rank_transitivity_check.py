#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Tests for the functionalities in the tranistivity_check file
"""


# =============================================================================
# IMPORTS
# =============================================================================

# import networkx as nx

import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg.simple import WeightedSumModel
from skcriteria.agg.topsis import TOPSIS
from skcriteria.agg import RankResult
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing import invert_objectives
from skcriteria.preprocessing import weighters
from skcriteria.preprocessing.filters import FilterGE, FilterNonDominated
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler
from skcriteria.ranksrev.rank_transitivity_check import (
    RankTransitivityChecker,
)
from skcriteria.cmp import RanksComparator, mkrank_cmp


# =============================================================================
# PROPERTIES
# =============================================================================


def test_RankTransitivityChecker():
    pipe = mkpipe(
        FilterGE({"ROE": 2}),  # Almenos rendir 2%,
        FilterNonDominated(),  # chau dominadas!
        InvertMinimize(),  # no más minimización!
        SumScaler(target="weights"),  # normalizamos los pesos
        SumScaler(target="matrix"),  # normalizamos la matriz
        WeightedSumModel(),  # Función de agregación
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
                "Recomposition.1",
                RankResult(
                    method="WeightedSumModel",
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
        FilterGE({"ROE": 2}),  # Almenos rendir 2%,
        FilterNonDominated(),  # chau dominadas!
        InvertMinimize(),  # no más minimización!
        SumScaler(target="weights"),  # normalizamos los pesos
        SumScaler(target="matrix"),  # normalizamos la matriz
        WeightedSumModel(),  # Función de agregación
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=False)

    dm = skc.datasets.load_simple_stock_selection()

    with pytest.raises(ValueError):
        dec.evaluate(dm)


def test_RankTransitivityChecker_vanheerden():
    # pipeline de la tesis entera
    pipe = mkpipe(
        invert_objectives.InvertMinimize(),
        weighters.EntropyWeighter(),
        SumScaler(target="matrix"),
        TOPSIS(),
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=True)

    dm = skc.datasets.load_van2021evaluation()

    result = dec.evaluate(dm)

    


def test_RankTransitivityChecker_bad_pipe():
    bad_pipe = "Suffering and pain"
    with pytest.raises(TypeError) as ex:
        RankTransitivityChecker(bad_pipe)
