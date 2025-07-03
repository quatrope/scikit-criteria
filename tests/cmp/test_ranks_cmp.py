#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.cmp.ranks_cmp"""


# =============================================================================
# IMPORTS
# =============================================================================

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pytest

import seaborn as sns

from skcriteria import agg
from skcriteria.cmp import ranks_cmp
from skcriteria.utils import Bunch


# =============================================================================
# TESTS
# =============================================================================


def test_RanksComparator_extra_not_mapping():
    rank = agg.RankResult("test", ["a"], [1], {})
    with pytest.raises(TypeError):
        ranks_cmp.RanksComparator([("a", rank), (1, rank)], extra=None)


def test_RanksComparator_name_not_str():
    rank = agg.RankResult("test", ["a"], [1], {})
    with pytest.raises(ValueError):
        ranks_cmp.RanksComparator([("a", rank), (1, rank)], extra={})


def test_RanksComparator_not_rank_result():
    rank = agg.RankResult("test", ["a"], [1], {})
    with pytest.raises(TypeError):
        ranks_cmp.RanksComparator([("a", rank), ("b", None)], extra={})


def test_RanksComparator_duplicated_names():
    rank = agg.RankResult("test", ["a"], [1], {})
    with pytest.raises(ValueError):
        ranks_cmp.RanksComparator([("a", rank), ("a", rank)], extra={})


def test_RanksComparator_missing_alternatives():
    rank0 = agg.RankResult("test", ["a"], [1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 2], {})
    with pytest.raises(ValueError):
        ranks_cmp.mkrank_cmp(rank0, rank1)


def test_RanksComparator_repr():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)
    assert repr(rcmp) == "<RanksComparator [ranks=['test_1', 'test_2']]>"


def test_RanksComparator_extra_is_None():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1, extra=None)
    assert rcmp.extra_ == {}


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_to_dataframe(untied):
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    df = ranks_cmp.mkrank_cmp(rank0, rank1).to_dataframe(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"a": 1, "b": 2 if untied else 1},
            "test_2": {"a": 1, "b": 2 if untied else 1},
        }
    )

    expected.columns.name = "Method"
    expected.index.name = "Alternatives"

    pd.testing.assert_frame_equal(df, expected)


def test_RanksComparator_extra():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)
    actual = rcmp.e_
    assert actual == Bunch("extra", {})


def test_RanksComparator_diff():
    rcmp = ranks_cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    rcmp_equal = ranks_cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    diff = rcmp.diff(rcmp_equal)
    assert diff.has_differences is False


def test_RanksComparator_diff_different_ranks_names():
    rcmp = ranks_cmp.RanksComparator(
        [
            ("r0", agg.RankResult("test", ["a", "b"], [1, 1], {})),
            ("r1", agg.RankResult("test", ["a", "b"], [1, 1], {})),
        ],
        extra={},
    )
    rcmp_different_rank = ranks_cmp.RanksComparator(
        [
            ("r0", agg.RankResult("test", ["a", "b"], [1, 1], {})),
            ("r2", agg.RankResult("test", ["a", "b"], [1, 1], {})),
        ],
        extra={},
    )
    diff = rcmp.diff(rcmp_different_rank)
    assert diff.has_differences


def test_RanksComparator_diff_different_ranks():
    rcmp = ranks_cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    rcmp_different_rank = ranks_cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 2], {}),
    )
    diff = rcmp.diff(rcmp_different_rank)
    assert diff.has_differences


def test_RanksComparator_diff_different_length():
    rcmp = ranks_cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
    )
    rcmp_three_ranks = ranks_cmp.mkrank_cmp(
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 1], {}),
        agg.RankResult("test", ["a", "b"], [1, 2], {}),
    )
    diff = rcmp.diff(rcmp_three_ranks)
    assert diff.has_differences


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_cov(untied):
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    cov = ranks_cmp.mkrank_cmp(rank0, rank1).cov(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": (
                {"test_1": 0.5, "test_2": 0.5}
                if untied
                else {"test_1": 0.0, "test_2": 0.0}
            ),
            "test_2": (
                {"test_1": 0.5, "test_2": 0.5}
                if untied
                else {"test_1": 0.0, "test_2": 0.0}
            ),
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(cov, expected)


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_corr(untied):
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    corr = ranks_cmp.mkrank_cmp(rank0, rank1).corr(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": (
                {"test_1": 1.0, "test_2": 1.0}
                if untied
                else {"test_1": np.nan, "test_2": np.nan}
            ),
            "test_2": (
                {"test_1": 1.0, "test_2": 1.0}
                if untied
                else {"test_1": np.nan, "test_2": np.nan}
            ),
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(corr, expected)


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_r2_score(untied):
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    r2 = ranks_cmp.mkrank_cmp(rank0, rank1).r2_score(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 1.0, "test_2": 1.0},
            "test_2": {"test_1": 1.0, "test_2": 1.0},
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(r2, expected)


@pytest.mark.parametrize("untied", [True, False])
def test_RanksComparator_distance(untied):
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    dis = ranks_cmp.mkrank_cmp(rank0, rank1).distance(untied=untied)

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 0.0, "test_2": 0.0},
            "test_2": {"test_1": 0.0, "test_2": 0.0},
        },
    )

    expected.columns.name = "Method"
    expected.index.name = "Method"

    pd.testing.assert_frame_equal(dis, expected)


def test_RanksComparator_len():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    assert len(ranks_cmp.mkrank_cmp(rank0, rank1)) == 2


def test_RanksComparator_getitem():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)
    copy = rcmp[0:]

    assert rank0 == rcmp["test_1"] == rcmp[0] == copy[0]
    assert rank1 == rcmp["test_2"] == rcmp[1] == copy[1]

    with pytest.raises(ValueError):
        rcmp[0::2]

    with pytest.raises(KeyError):
        rcmp[object]


def test_RanksComparator_hash():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)
    assert id(rcmp) == hash(rcmp)


def test_RanksComparator_extra_get():
    rank0 = agg.RankResult(
        "test", ["a", "b"], [1, 1], {"alpha": 1, "bravo": 2}
    )
    rank1 = agg.RankResult(
        "test", ["a", "b"], [1, 1], {"alpha": 1, "delta": 3}
    )
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    assert rcmp.extra_get("alpha") == {"test_1": 1, "test_2": 1}
    assert rcmp.extra_get("bravo") == {"test_1": 2, "test_2": None}
    assert rcmp.extra_get("delta", "foo") == {"test_1": "foo", "test_2": 3}
    assert rcmp.extra_get("charly", "foo") == {
        "test_1": "foo",
        "test_2": "foo",
    }


def test_RanksComparator_plot():
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

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

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.flow(ax=test_ax, untied=untied)

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

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.reg(ax=test_ax, untied=untied)

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
    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    with pytest.raises(TypeError):
        rcmp.plot.reg(color="k", untied=untied)


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_heatmap(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.heatmap(ax=test_ax, untied=untied)

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

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.corr(ax=test_ax, untied=untied)

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

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.cov(ax=test_ax, untied=untied)

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
@check_figures_equal()
def test_RanksComparatorPlotter_r2_score(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.r2_score(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 1.0, "test_2": 1.0},
            "test_2": {"test_1": 1.0, "test_2": 1.0},
        },
    )
    expected.columns.name = "Method"
    expected.index.name = "Method"

    sns.heatmap(
        expected,
        annot=True,
        cbar_kws={"label": "$R^2$"},
        ax=exp_ax,
    )


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@check_figures_equal()
def test_RanksComparatorPlotter_distance(fig_test, fig_ref, untied):
    test_ax = fig_test.subplots()

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.distance(ax=test_ax, untied=untied)

    # EXPECTED
    exp_ax = fig_ref.subplots()

    expected = pd.DataFrame.from_dict(
        {
            "test_1": {"test_1": 0, "test_2": 0},
            "test_2": {"test_1": 0, "test_2": 0},
        },
    )
    expected.columns.name = "Method"
    expected.index.name = "Method"

    sns.heatmap(
        expected,
        annot=True,
        cbar_kws={"label": "Hamming distance"},
        ax=exp_ax,
    )


@pytest.mark.slow
@pytest.mark.parametrize("untied", [True, False])
@pytest.mark.parametrize("orient", ["v", "h"])
@check_figures_equal()
def test_RanksComparatorPlotter_box(fig_test, fig_ref, untied, orient):
    test_ax = fig_test.subplots()

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.box(ax=test_ax, orient=orient, untied=untied)

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

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.bar(ax=test_ax, untied=untied)

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

    rank0 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rank1 = agg.RankResult("test", ["a", "b"], [1, 1], {})
    rcmp = ranks_cmp.mkrank_cmp(rank0, rank1)

    rcmp.plot.barh(ax=test_ax, untied=untied)

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
