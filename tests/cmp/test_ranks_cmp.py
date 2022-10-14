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

import pandas as pd

import pytest

import seaborn as sns

from skcriteria import madm
from skcriteria.cmp import ranks_cmp

# =============================================================================
# TESTS
# =============================================================================


def test_ranks_merge_only_one_rank():
    rank = madm.RankResult("test", ["a"], [1], {})
    with pytest.raises(ValueError):
        ranks_cmp.ranks.merge([rank])


def test_ranks_merge_only_missing_alternatives():
    rank0 = madm.RankResult("test", ["a"], [1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 2], {})
    with pytest.raises(ValueError):
        ranks_cmp.ranks.merge([rank0, rank1])


def test_ranks_repr():
    assert repr(ranks_cmp.ranks) == "_RanksComparator"


@pytest.mark.parametrize("untied", [True, False])
def test_ranks_merge(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    df = ranks_cmp.ranks.merge([rank0, rank1], untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )

    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    pd.testing.assert_frame_equal(df, expected)


def test_ranks_merge_with_alias():
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    df = ranks_cmp.ranks.merge({"a0": rank0, "a1": rank1})

    expected = pd.DataFrame.from_dict(
        {"a0": {"a": 1, "b": 1}, "a1": {"a": 1, "b": 1}}
    )

    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_ranks_flow(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.flow([rank0, rank1], ax=test_ax, untied=untied)

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
def test_ranks_reg(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.reg([rank0, rank1], ax=test_ax, untied=untied)

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
        label="x=test_1, y=test_2 - R2=1",
        ax=exp_ax,
    )

    ranks_label = ranks_cmp.RANKS_LABELS[untied]
    exp_ax.set(xlabel=f"'x' {ranks_label}", ylabel=f"'y' {ranks_label}")

    exp_ax.legend()


@pytest.mark.parametrize("untied", [True, False])
def test_ranks_reg_unexpected_keyword_argument_color(untied):
    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    with pytest.raises(TypeError):
        ranks_cmp.ranks.reg([rank0, rank1], color="k", untied=untied)


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_ranks_heatmap(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.heatmap([rank0, rank1], ax=test_ax, untied=untied)

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
def test_ranks_corr(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.corr([rank0, rank1], ax=test_ax, untied=untied)

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
def test_ranks_cov(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.cov([rank0, rank1], ax=test_ax, untied=untied)

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
def test_ranks_box(fig_test, fig_ref, untied, orient):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.box(
        [rank0, rank1], ax=test_ax, orient=orient, untied=untied
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
def test_ranks_bar(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.bar([rank0, rank1], ax=test_ax, untied=untied)

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
def test_ranks_barh(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 1], {})
    ranks_cmp.ranks.barh([rank0, rank1], ax=test_ax, untied=untied)

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
