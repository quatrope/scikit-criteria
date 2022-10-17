#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for ``skcriteria.datasets``.

"""


# =============================================================================
# IMPORTS
# =============================================================================


import skcriteria as skc
from skcriteria import datasets


# =============================================================================
# TESTS
# =============================================================================


def test_load_simple_stock_selection():
    df = datasets.load_simple_stock_selection()
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
    assert df.equals(expected)
