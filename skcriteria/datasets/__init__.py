#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The :mod:`skcriteria.datasets` module includes utilities to load \
datasets."""


# =============================================================================
# IMPORRTS
# =============================================================================

from .. import core

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_simple_stock_selection():
    """Simple stock selection decision matrix.

    This matrix was designed primarily for teaching and evaluating the behavior
    of an experiment.

    Among the data we can find: two maximization criteria (ROE, CAP),
    one minimization criterion (RI), dominated alternatives (FX), and
    one alternative with an outlier criterion (ROE, MM = 1).

    Although the criteria and alternatives are original from the authors of
    Scikit-Criteria, the numerical values were extracted at some point from a
    somewhere which we have forgotten.

    Description:

    In order to decide to buy a series of stocks, a company studied 5 candidate
    investments: PE, JN, AA, FX, MM and GN. The finance department decides to
    consider the following criteria for selection:

        1. ROE (Max): Return % for each monetary unit invested.
        2. CAP (Max): Years of market capitalization.
        3. RI (Min): Risk of the stock.

    The decision maker determined that ROE has a importance of 2, CAP 4 and
    RI 1.

    """
    dm = core.mkdm(
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
    return dm
