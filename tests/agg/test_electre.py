#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.electre."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria
from skcriteria.agg.electre import (
    ELECTRE1,
    ELECTRE2,
    _electre2_ranker,
    concordance,
    discordance,
    electre2,
    weights_outrank,
)
from skcriteria.preprocessing.scalers import SumScaler, scale_by_sum

# =============================================================================
# TESTS
# =============================================================================


def test_concordance_cebrian2009localizacion():
    """
    Data From:
        Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        en Aragón: Una aplicación empírica de la ayuda a la decisión
        multicriterio tipo ELECTRE I y III. Robustez de los resultados
        obtenidos.
        Revista de Métodos Cuantitativos para la Economía y la Empresa,
        (7), 31-56.

    """

    matrix = scale_by_sum(
        [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        axis=0,
    )
    objectives = [1, 1, -1, 1, 1]
    weights = [0.25, 0.25, 0.1, 0.2, 0.2]

    expected = [
        [np.nan, 0.5000, 0.3500, 0.5000, 0.3500, 0.4500],
        [0.5000, np.nan, 0.5000, 0.7500, 0.5000, 0.5000],
        [0.6500, 0.5000, np.nan, 0.4500, 0.2000, 0.7000],
        [0.7500, 0.2500, 0.5500, np.nan, 0.3500, 0.4500],
        [0.9000, 0.7000, 0.8000, 0.9000, np.nan, 0.9000],
        [0.5500, 0.5000, 0.5500, 0.5500, 0.1000, np.nan],
    ]

    result = concordance(matrix, objectives, weights)

    assert np.allclose(result, expected, atol=1.0e-3, equal_nan=True)


def test_discordance_cebrian2009localizacion():
    """
    Data From:
        Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        en Aragón: Una aplicación empírica de la ayuda a la decisión
        multicriterio tipo ELECTRE I y III. Robustez de los resultados
        obtenidos.
        Revista de Métodos Cuantitativos para la Economía y la Empresa,
        (7), 31-56.

    """
    matrix = scale_by_sum(
        [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        axis=0,
    )
    objectives = [1, 1, -1, 1, 1]
    expected = [
        [np.nan, 0.5052, 0.4042, 0.1883, 0.42857, 0.2825],
        [0.4286, np.nan, 0.7143, 0.2589, 0.8571, 0.5714],
        [0.1696, 0.2825, np.nan, 0.1938, 0.2825, 0.2180],
        [0.5714, 0.4042, 0.8571, np.nan, 1.0, 0.7143],
        [0.0485, 0.3031, 0.2021, 0.0727, np.nan, 0.0969],
        [0.1295, 0.6063, 0.5052, 0.2021, 0.3031, np.nan],
    ]

    results = discordance(matrix, objectives)

    assert np.allclose(results, expected, atol=1.0e-3, equal_nan=True)


def test_ELECTRE1_cebrian2009localizacion():
    """
    Data From:
        Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        en Aragón: Una aplicación empírica de la ayuda a la decisión
        multicriterio tipo ELECTRE I y III. Robustez de los resultados
        obtenidos.
        Revista de Métodos Cuantitativos para la Economía y la Empresa,
        (7), 31-56.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        objectives=[1, 1, -1, 1, 1],
        weights=[0.25, 0.25, 0.1, 0.2, 0.2],
    )

    scaler = SumScaler("both")
    dm = scaler.transform(dm)

    kselector = ELECTRE1()
    result = kselector.evaluate(dm)

    assert np.all(result.kernel_where_ == [4])


def test_ELECTRE1_kernel_sensibility_barba1997decisiones():
    """
    Data From:
        Barba-Romero, S., & Pomerol, J. C. (1997).
        Decisiones multicriterio: fundamentos teóricos y utilización
        práctica. P.216
    """
    dm = skcriteria.mkdm(
        matrix=[
            [0.188, 0.172, 0.168, 0.122, 0.114],
            [0.125, 0.069, 0.188, 0.244, 0.205],
            [0.156, 0.241, 0.134, 0.220, 0.136],
            [0.188, 0.034, 0.174, 0.146, 0.159],
            [0.188, 0.276, 0.156, 0.171, 0.205],
            [0.156, 0.207, 0.180, 0.098, 0.182],
        ],
        weights=[0.25, 0.25, 0.10, 0.20, 0.20],
        objectives=[1, 1, 1, 1, 1],
        alternatives=["A", "B", "D", "E", "G", "H"],
    )

    ps = [0.50, 0.60, 0.70, 0.80, 0.89, 0.89, 0.89, 0.94, 1]
    qs = [0.50, 0.40, 0.30, 0.20, 0.10, 0.08, 0.05, 0.05, 0]

    scaler = SumScaler("both")
    dm = scaler.transform(dm)

    kernels_len = []
    for p, q in zip(ps, qs):
        kselector = ELECTRE1(p=p, q=q)
        dec = kselector.evaluate(dm)
        klen = len(dec.kernel_)

        if kernels_len and klen < kernels_len[-1]:
            pytest.fail(
                "less sensitive electre must have more "
                f"alternatives in kernel. {kernels_len}"
            )
        kernels_len.append(klen)


def test_ELECTRE1_invalid_p_and_q():
    with pytest.raises(ValueError):
        ELECTRE1(p=1.5)

    with pytest.raises(ValueError):
        ELECTRE1(q=10)

    with pytest.raises(ValueError):
        ELECTRE1(p=-1)

    with pytest.raises(ValueError):
        ELECTRE1(q=-1)


# =============================================================================
# ELECTRE II
# =============================================================================


def test_weight_outrank():
    matrix = scale_by_sum(
        [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        axis=0,
    )
    objectives = [1, 1, -1, 1, 1]
    weights = [0.25, 0.25, 0.1, 0.2, 0.2]
    expected = [
        [False, True, True, True, True, True],
        [False, False, False, False, True, False],
        [False, True, False, True, True, True],
        [False, True, False, False, True, True],
        [False, True, False, False, False, False],
        [False, True, True, False, True, False],
    ]

    results = weights_outrank(matrix, objectives, weights)
    np.testing.assert_array_equal(results, expected)


def test_electre_2_ranker_empty_kernel():
    outrank_s = np.array(
        [
            [False, True, False],  # Alternative 0 outranks Alternative 1
            [False, False, True],  # Alternative 1 outranks Alternative 2
            [True, False, False],  # Alternative 2 outranks Alternative 0
        ]
    )

    # Weak outranking matrix - everyone also weakly outranks someone else
    outrank_w = np.array(
        [
            [
                False,
                False,
                True,
            ],  # Alternative 0 weakly outranks Alternative 2
            [
                True,
                False,
                False,
            ],  # Alternative 1 weakly outranks Alternative 0
            [
                False,
                True,
                False,
            ],  # Alternative 2 weakly outranks Alternative 1
        ]
    )

    _electre2_ranker(3, outrank_s, outrank_w, invert_ranking=False)


def test_ELECTRE2_cebrian2009localizacion():
    """
    Data From:
        Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        en Aragón: Una aplicación empírica de la ayuda a la decisión
        multicriterio tipo ELECTRE I y III. Robustez de los resultados
        obtenidos.
        Revista de Métodos Cuantitativos para la Economía y la Empresa,
        (7), 31-56.

    """
    dm = skcriteria.mkdm(
        matrix=[
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        objectives=[1, 1, -1, 1, 1],
        weights=[0.25, 0.25, 0.1, 0.2, 0.2],
    )

    scaler = SumScaler("both")
    dm = scaler.transform(dm)

    kselector = ELECTRE2()
    result = kselector.evaluate(dm)

    assert np.all(result.rank_ == [6, 3, 2, 5, 1, 4])
    np.testing.assert_allclose(result.e_.score, [5.0, 2.5, 2.0, 4.0, 1.0, 3.0])


def test_ELECTRE2_wang2006():
    """
    Data From:
        Xiaoting Wang, Evangelos Triantaphyllou,
            Ranking irregularities when evaluating alternatives by using
            some ELECTRE methods,
        Omega,
        Volume 36, Issue 1,
        2008,

    """
    dm = skcriteria.mkdm(
        matrix=[
            [1, 2, 1, 5, 2, 2, 4],
            [3, 5, 3, 5, 3, 3, 3],
            [3, 5, 3, 5, 3, 2, 2],
            [1, 2, 2, 5, 1, 1, 1],
            [1, 1, 3, 5, 4, 1, 5],
        ],
        alternatives=["A1", "A2", "A3", "A4", "A5"],
        objectives=[max, max, max, max, max, max, max],
        weights=[0.0780, 0.1180, 0.1570, 0.3140, 0.2350, 0.0390, 0.0590],
    )

    kselector = ELECTRE2()
    result = kselector.evaluate(dm)

    assert np.all(result.rank_ == [4, 1, 3, 5, 2])
    np.testing.assert_allclose(result.e_.score, [3.0, 1.0, 2.0, 4.0, 1.5])


@pytest.mark.parametrize(
    "p0, p1, p2", [(1, 2, 3), (0, 0.5, 1), (1, 0.5, 0.75), (-1, 0.5, 0.25)]
)
def test_ELECTRE2_invalid_ps(p0, p1, p2):
    with pytest.raises(ValueError):
        ELECTRE2(p0=p0, p1=p1, p2=p2)


@pytest.mark.parametrize("q0, q1", [(1, 2), (0, 0.5), (-1, 0.9)])
def test_ELECTRE2_invalid_qs(q0, q1):
    with pytest.raises(ValueError):
        ELECTRE2(q0=q0, q1=q1)


def test_electre2_deprecation():
    matrix = np.array(
        [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        dtype=float,
    )
    objectives = np.array([1, 1, -1, 1, 1])
    weights = np.array([0.25, 0.25, 0.1, 0.2, 0.2])

    with pytest.warns(DeprecationWarning):
        electre2(matrix, objectives, weights)


def test_weights_outrank_deprecation():
    matrix = np.array(
        [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8],
        ],
        dtype=float,
    )
    objectives = np.array([1, 1, -1, 1, 1])
    weights = np.array([0.25, 0.25, 0.1, 0.2, 0.2])

    with pytest.warns(DeprecationWarning):
        weights_outrank(matrix, objectives, weights)
