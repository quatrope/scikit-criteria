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
from skcriteria.madm import SIMUS
from skcriteria.preprocessing import VectorScaler

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
        anames=["Proyecto 1", "Proyecto 2", "Proyecto 3"],
        cnames=["Criterio 1", "Criterio 2", "Criterio 3", "Criterio 4"],
    )
    b = [None, 500, None, None]

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

    ranker = SIMUS()
    result = ranker.rank(dm, b=b)

    # assert result.equals(expected)
    # assert result.method == expected.method
    # assert np.all(result.e_.ideal == expected.e_.ideal)
    # assert np.allclose(result.e_.anti_ideal, expected.e_.anti_ideal)
    # assert np.allclose(result.e_.similarity, expected.e_.similarity)
