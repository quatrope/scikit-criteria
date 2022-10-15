#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.cmp.ranks_cmp

"""


# =============================================================================
# IMPORTS
# =============================================================================

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pytest

import seaborn as sns

from skcriteria import madm
from skcriteria.cmp import ranks_cmp

# =============================================================================
# TESTS
# =============================================================================


def test_Ranks_only_one_rank():
    rank = madm.RankResult("test", ["a"], [1], {})
    with pytest.raises(ValueError):
        ranks_cmp.RanksComparator([rank])


def test_RanksComparator_missing_alternatives():
    rank0 = madm.RankResult("test", ["a"], [1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 2], {})
    with pytest.raises(ValueError):
        ranks_cmp.RanksComparator([rank0, rank1])


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_to_dataframe(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    df = ranks_cmp.RanksComparator([rank0, rank1]).to_dataframe(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )

    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    pd.testing.assert_frame_equal(df, expected)


def test_RanksComparator_to_dataframe_with_alias():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    df = ranks_cmp.RanksComparator({"a0": rank0, "a1": rank1}).to_dataframe()

    expected = pd.DataFrame.from_dict(
        {"a0": {"a": 1, "b": 1}, "a1": {"a": 1, "b": 1}}
    )

    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_cov(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    cov = ranks_cmp.RanksComparator([rank0, rank1]).cov(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 0.5, "test_2": 0.5}
            if untied
            else {"test_1": 0.0, "test_2": 0.0},
            "test_2": {"test_1": 0.5, "test_2": 0.5}
            if untied
            else {"test_1": 0.0, "test_2": 0.0},
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(cov, expected)


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_corr(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    corr = ranks_cmp.RanksComparator([rank0, rank1]).corr(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 1.0, "test_2": 1.0}
            if untied
            else {"test_1": np.nan, "test_2": np.nan},
            "test_2": {"test_1": 1.0, "test_2": 1.0}
            if untied
            else {"test_1": np.nan, "test_2": np.nan},
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(corr, expected)


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_r2_score(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    r2 = ranks_cmp.RanksComparator([rank0, rank1]).r2_score(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 1.0, "test_2": 1.0},
            "test_2": {"test_1": 1.0, "test_2": 1.0},
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(r2, expected)


def test_RanksComparator_repr():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    assert (
        repr(ranks_cmp.RanksComparator([rank0, rank1]))
        == "<RanksComparator ranks=['test_1', 'test_2']>"
    )


def test_RanksComparator_len():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    assert len(ranks_cmp.RanksComparator([rank0, rank1])) == 2


def test_RanksComparator_iter():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    nr0, nr1 = iter(ranks_cmp.RanksComparator([rank0, rank1]))
    assert nr0 == "test_1"
    assert nr1 == "test_2"


def test_RanksComparator_getattr():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])
    assert rank0 == rcmp["test_1"]
    assert rank1 == rcmp["test_2"]


def test_RanksComparator_hash():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])
    assert id(rcmp) == hash(rcmp)


def test_RanksComparator_plot():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    assert isinstance(rcmp.plot, ranks_cmp.RanksComparatorPlotter)
    assert rcmp.plot._ranks_cmp is rcmp


# =============================================================================
# RanksComparatorPlotter
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_flow(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).flow(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    sns.lineplot(data=expected.T, estimator=None, sort=False, ax=exp_ax)
    exp_ax.grid(alpha=0.3)

    exp_ax.set_ylabel(ranks_cmp.RANKS_LABELS[untied])


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_reg(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).reg(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    sns.regplot(
        x="test_1",
        y="test_2",
        data=expected,
        label="x=test_1, y=test_2 - $R^2=1$",
        ax=exp_ax,
    )

    ranks_label = ranks_cmp.RANKS_LABELS[untied]
    exp_ax.set(xlabel=f"'x' {ranks_label}", ylabel=f"'y' {ranks_label}")

    exp_ax.legend()


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparatorPlotter_reg_unexpected_keyword_argument_color(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    with pytest.raises(TypeError):
        ranks_cmp.RanksComparatorPlotter(rcmp).reg(color="k", untied=untied)


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_heatmap(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).heatmap(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    sns.heatmap(
        expected,
        annot=True,
        cbar_kws={"label": ranks_cmp.RANKS_LABELS[untied]},
        ax=exp_ax,
    )


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_corr(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).corr(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    sns.heatmap(
        expected.corr(),
        annot=True,
        cbar_kws={"label": "Correlation"},
        ax=exp_ax,
    )


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_cov(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).cov(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    sns.heatmap(
        expected.cov(),
        annot=True,
        cbar_kws={"label": "Covariance"},
        ax=exp_ax,
    )


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@pytest.mark.parametrize("orient", ["v", "h"])
@check_figures_equal()
def test_RanksComparatorPlotter_box(fig_test, fig_ref, untied, orient):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).box(
        ax=test_ax, orient=orient, untied=untied
    )

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    sns.boxplot(data=expected.T, orient=orient)

    ranks_label = ranks_cmp.RANKS_LABELS[untied]
    if orient in (None, "v"):
        exp_ax.set_ylabel(ranks_label)
    else:
        exp_ax.set_xlabel(ranks_label)


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_bar(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).bar(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    expected.plot.bar(ax=exp_ax)

    exp_ax.set_ylabel(ranks_cmp.RANKS_LABELS[untied])


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_barh(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.RanksComparator([rank0, rank1])

    ranks_cmp.RanksComparatorPlotter(rcmp).barh(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )
    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    expected.plot.barh(ax=exp_ax)

    exp_ax.set_xlabel(ranks_cmp.RANKS_LABELS[untied])
