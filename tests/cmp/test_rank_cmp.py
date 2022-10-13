from matplotlib import pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import pandas as pd

import pytest

import seaborn as sns

from skcriteria.cmp import ranks_cmp
from skcriteria import madm


def test_ranks_merge_only_one_rank():
    rank = madm.RankResult("test", ["a"], [1], {})
    with pytest.raises(ValueError):
        ranks_cmp.ranks.merge([rank])


def test_ranks_merge_only_missing_alternatives():
    rank0 = madm.RankResult("test", ["a"], [1], {})
    rank1 = madm.RankResult("test", ["a", "b"], [1, 2], {})
    with pytest.raises(ValueError):
        ranks_cmp.ranks.merge([rank0, rank1])


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

    sns.lineplot(data=expected.T, estimator=None, sort=False, ax=exp_ax)
    exp_ax.grid(alpha=0.3)

    handles, labels = exp_ax.get_legend_handles_labels()
    exp_ax.legend(handles, labels, title="Alternatives")

    exp_ax.set_xlabel("Method")
    exp_ax.set_ylabel(ranks_cmp._RANKS_LABELS[untied])
    exp_ax.invert_yaxis()


