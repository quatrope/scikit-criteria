#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for ``skcriteria.datasets``."""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

import skcriteria as skc
from skcriteria import datasets


# =============================================================================
# Tsimple_stock_selection
# =============================================================================


def test_load_simple_stock_selection():
    dm = datasets.load_simple_stock_selection()
    expected = skc.mkdm(
        matrix=[
            [7, 5, 35],
            [5, 4, 26],
            [5, 6, 28],
            [3, 4, 36],
            [1, 7, 30],
            [5, 8, 30],
        ],
        objectives=[max, max, min],
        weights=[2, 4, 1],
        alternatives=["PE", "JN", "AA", "FX", "MM", "GN"],
        criteria=["ROE", "CAP", "RI"],
    )
    assert dm.equals(expected)


# =============================================================================
# load_van2021evaluation
# =============================================================================


def test_load_van2021evaluation_windows_size_invalid():
    with pytest.raises(ValueError):
        datasets.load_van2021evaluation(windows_size=17)


def test_load_van2021evaluation_windows_size_7():
    dm = datasets.load_van2021evaluation(windows_size=7)
    expected = skc.mkdm(
        matrix=[
            [0.029, 0.156, 8144000000.0, 15860000000.0, 0.312, 1.821e-11],
            [0.033, 0.167, 6141000000.0, 11180000000.0, 0.396, 9.167e-09],
            [0.015, 0.097, 209500000000.0, 138800000000.0, 0.281, 1.254e-08],
            [0.057, 0.399, 8287000000.0, 27260000000.0, 0.327, 1.459e-12],
            [0.023, 0.127, 100000000000.0, 80540000000.0, 0.313, 1.737e-09],
            [0.04, 0.179, 6707000000.0, 16650000000.0, 0.319, 1.582e-09],
            [0.015, 0.134, 25130000000.0, 17310000000.0, 0.32, 1.816e-09],
            [0.013, 0.176, 4157000000.0, 5469000000.0, 0.321, 1.876e-11],
            [0.014, 0.164, 23080000000.0, 29240000000.0, 0.322, 7.996e-12],
        ],
        objectives=[max, min, max, min, max, max],
        alternatives=[
            "ADA",
            "BNB",
            "BTC",
            "DOGE",
            "ETH",
            "LINK",
            "LTC",
            "XLM",
            "XRP",
        ],
        criteria=["xRV", "sRV", "xVV", "sVV", "xR2", "xm"],
    )
    assert dm.equals(expected)


def test_load_van2021evaluation_windows_size_15():
    dm = datasets.load_van2021evaluation(windows_size=15)
    expected = skc.mkdm(
        matrix=[
            [0.072, 0.274, 17440000000.0, 32880000000.0, 0.281, 3.806e-11],
            [0.087, 0.348, 13160000000.0, 23330000000.0, 0.339, 1.195e-08],
            [0.036, 0.159, 450200000000.0, 289400000000.0, 0.237, 2.192e-08],
            [0.153, 0.805, 17770000000.0, 52850000000.0, 0.314, 2.441e-12],
            [0.055, 0.213, 214500000000.0, 169500000000.0, 0.239, 2.52e-09],
            [0.097, 0.302, 14440000000.0, 27920000000.0, 0.277, 2.544e-09],
            [0.034, 0.207, 54150000000.0, 35570000000.0, 0.28, 2.679e-09],
            [0.031, 0.275, 8951000000.0, 11040000000.0, 0.276, 2.454e-11],
            [0.037, 0.292, 49660000000.0, 59500000000.0, 0.26, 9.236e-12],
        ],
        objectives=[max, min, max, min, max, max],
        alternatives=[
            "ADA",
            "BNB",
            "BTC",
            "DOGE",
            "ETH",
            "LINK",
            "LTC",
            "XLM",
            "XRP",
        ],
        criteria=["xRV", "sRV", "xVV", "sVV", "xR2", "xm"],
    )
    assert dm.equals(expected)
