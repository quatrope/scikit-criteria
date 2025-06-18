#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.core.stats"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd


import pytest

from skcriteria.core import data


# =============================================================================
# STATS
# =============================================================================


def test_DecisionMatrixStatsAccessor_default_kind(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    stats = data.DecisionMatrixStatsAccessor(dm)

    assert stats().equals(dm._data_df.describe())


@pytest.mark.parametrize(
    "kind", data.DecisionMatrixStatsAccessor._DF_WHITELIST
)
def test_DecisionMatrixStatsAccessor_df_whitelist_by_kind(
    kind, decision_matrix
):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    expected = getattr(dm._data_df, kind)()

    stats = data.DecisionMatrixStatsAccessor(dm)

    result_call = stats(kind=kind)

    def cmp(r, e):
        return (
            r.equals(e)
            if isinstance(result_call, (pd.DataFrame, pd.Series))
            else np.equal
        )

    result_method = getattr(stats, kind)()

    assert cmp(result_call, expected)
    assert cmp(result_method, expected)


def test_DecisionMatrixStatsAccessor_mad(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    # Expected
    df = dm._data_df
    expected = (df - df.mean()).abs().mean()

    # result
    stats = data.DecisionMatrixStatsAccessor(dm)
    result = stats(kind="mad")

    assert result.equals(expected)


def test_DecisionMatrixStatsAccessor_invalid_kind(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    stats = data.DecisionMatrixStatsAccessor(dm)

    with pytest.raises(ValueError):
        stats("_dm")

    stats.foo = None
    with pytest.raises(ValueError):
        stats("foo")

    with pytest.raises(AttributeError):
        stats.to_csv()


def test_DecisionMatrixStatsAccessor_dir(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    stats = data.DecisionMatrixStatsAccessor(dm)

    expected = set(data.DecisionMatrixStatsAccessor._DF_WHITELIST)
    result = dir(stats)

    assert not expected.difference(result)
