#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.simus."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg import RankResult
from skcriteria.agg.simus import SIMUS

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_SIMUS_munier24metodo():
    """
    Data From:
        Munier, N., Carignano, C., & Alberto, C.
        UN MÉTODO DE PROGRAMACIÓN MULTIOBJETIVO.
        Revista de la Escuela de Perfeccionamiento en Investigación
        Operativa, 24(39).
    """

    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
        alternatives=["Proyecto 1", "Proyecto 2", "Proyecto 3"],
        criteria=["Criterio 1", "Criterio 2", "Criterio 3", "Criterio 4"],
    )
    b = [None, 500, None, None]

    expected = RankResult(
        "SIMUS",
        ["Proyecto 1", "Proyecto 2", "Proyecto 3"],
        [3, 2, 1],
        {
            "method_1_score": [0.09090909, 0.66603535, 0.74305556],
            "method_2_score": [-2.45454545, 0.99621211, 1.45833334],
            "rank_by": 1,
        },
    )

    ranker = SIMUS()
    result = ranker.evaluate(dm, b=b)

    assert result.values_equals(expected)
    assert result.method == expected.method
    assert np.all(result.e_.b == b)
    assert np.all(result.e_.rank_by == expected.e_.rank_by)
    assert np.allclose(result.e_.method_1_score, expected.e_.method_1_score)
    assert np.allclose(result.e_.method_2_score, expected.e_.method_2_score)


def test_SIMUS_solver_not_available():
    with pytest.raises(ValueError):
        SIMUS(solver="fooo")


def test_SIMUS_solver_incorrect_rank_by():
    with pytest.raises(ValueError):
        SIMUS(rank_by=3)


def test_SIMUS_multiple_weights_warning():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
        weights=[1, 2, 3, 4],
    )

    ranker = SIMUS()
    with pytest.warns(UserWarning):
        ranker.evaluate(dm)


def test_SIMUS_incorrect_b():
    dm = skcriteria.mkdm(
        matrix=[
            [250, 120, 20, 800],
            [130, 200, 40, 1000],
            [350, 340, 15, 600],
        ],
        objectives=[max, max, min, max],
    )
    b = [1, 2, 3]

    ranker = SIMUS()
    with pytest.raises(ValueError):
        ranker.evaluate(dm, b=b)
