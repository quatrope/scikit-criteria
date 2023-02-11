#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.cmanagers

"""


# =============================================================================
# IMPORTS
# =============================================================================


import pandas as pd


from skcriteria.utils import cmanagers


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_df_temporal_header():
    df = pd.DataFrame({"x": [1], "y": [2]})
    df.columns.name = "original"

    with cmanagers.df_temporal_header(df, ["a", "b"], "replaced") as df:
        pd.testing.assert_index_equal(
            df.columns, pd.Index(["a", "b"], name="replaced")
        )

    pd.testing.assert_index_equal(
        df.columns, pd.Index(["x", "y"], name="original")
    )
