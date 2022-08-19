#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.distance

"""

# =============================================================================
# IMPORTS
# =============================================================================


import skcriteria as skc
from skcriteria.preprocessing.distance import CenitDistance


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_CenitDistance_simple_matrix():

    dm = skc.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = skc.mkdm(
        matrix=[[-0.0, 0.0, 1.0], [1.0, 1.0, -0.0]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    tfm = CenitDistance()

    result = tfm.transform(dm)

    assert result.equals(expected)


def test_CenitDistance_diakoulaki1995determining():
    """
    Data from:
        Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        Determining objective weights in multiple criteria problems:
        The critic method. Computers & Operations Research, 22(7), 763-770.
    """

    dm = skc.mkdm(
        matrix=[
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98],
        ],
        objectives=[max, max, max],
        weights=[61, 1.08, 4.33],
    )

    expected = skc.mkdm(
        matrix=[
            [1.0, 0.21243523, 0.99702381],
            [0.57845188, 0.0, 1.0],
            [0.53242678, 0.44559585, 0.46130952],
            [0.45606695, 0.78497409, 0.19940476],
            [0.41841004, 0.65025907, 0.40178571],
            [0.40376569, 1.0, 0.06845238],
            [0.29811715, 0.84455959, 0.33333333],
            [0.0, 0.79015544, 0.0],
        ],
        objectives=[max, max, max],
        weights=[61, 1.08, 4.33],
    )

    tfm = CenitDistance()
    result = tfm.transform(dm)

    assert result.aequals(expected)


def test_CenitDistance_no_change_original_dm():

    dm = skc.mkdm(
        matrix=[[1, 0, 3], [0, 5, 6]],
        objectives=[min, max, min],
        weights=[1, 2, 0],
    )

    expected = dm.copy()

    tfm = CenitDistance()
    dmt = tfm.transform(dm)

    assert (
        dm.equals(expected) and not dmt.equals(expected) and dm is not expected
    )
