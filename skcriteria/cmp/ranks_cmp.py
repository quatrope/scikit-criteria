#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Ranking comparation utilities."""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import itertools as it
from collections import Counter, defaultdict
from collections.abc import Mapping

import matplotlib.pyplot as plt

import pandas as pd

from scipy.spatial import distance

import seaborn as sns

from sklearn import metrics as _skl_metrics

from ..utils import AccessorABC, unique_names


# =============================================================================
# CONSTANTS
# =============================================================================

RANKS_LABELS = {
    True: "Untied ranks (lower is better)",
    False: "Ranks (lower is better)",
}

# =============================================================================
# PLOTTER
# =============================================================================


class RanksComparatorPlotter(AccessorABC):

    _default_kind = "flow"

    def __init__(self, ranks_cmp):
        self._ranks_cmp = ranks_cmp

    # MANUAL MADE PLOT ========================================================
    # These plots have a much more manually orchestrated code.

    def flow(self, *, untied=False, grid_kws=None, **kwargs):
        df = self._ranks_cmp.to_dataframe(untied=untied)

        ax = sns.lineplot(data=df.T, estimator=None, sort=False, **kwargs)

        grid_kws = {} if grid_kws is None else grid_kws
        grid_kws.setdefault("alpha", 0.3)
        ax.grid(**grid_kws)

        ax.set_ylabel(RANKS_LABELS[untied])

        return ax

    def reg(
        self,
        *,
        untied=False,
        r2=True,
        palette=None,
        legend=True,
        r2_fmt=".2g",
        **kwargs,
    ):

        df = self._ranks_cmp.to_dataframe(untied=untied)

        # Just to ensure that no manual color reaches regplot
        if "color" in kwargs:
            cls_name = type(self).__name__
            raise TypeError(
                f"{cls_name}.reg() got an unexpected keyword argument 'color'"
            )

        # we create the infinite cycle of colors for the palette,
        # so we take out as we need
        colors = it.cycle(sns.color_palette(palette=palette))

        # if there is a custom axis, we take it out
        ax = kwargs.pop("ax", None)

        # r2
        if r2:
            r2_df = self._ranks_cmp.r2_score(untied=untied)

        # pairwise ranks iteration
        for x, y in it.combinations(df.columns, 2):
            color = next(colors)

            # The r2 correlation index
            r2_label = ""
            if r2:
                r2_score = format(r2_df[x][y], r2_fmt)
                r2_label = f" - $R^2={r2_score}$"

            label = "x={x}, y={y}{r2}".format(x=x, y=y, r2=r2_label)
            ax = sns.regplot(
                x=x, y=y, data=df, ax=ax, label=label, color=color, **kwargs
            )

        ranks_label = RANKS_LABELS[untied]
        ax.set(xlabel=f"'x' {ranks_label}", ylabel=f"'y' {ranks_label}")

        if legend:
            ax.legend()

        return ax

    # SEABORN BASED ===========================================================
    # Thin wrapper around seaborn plots

    def heatmap(self, *, untied=False, **kwargs):
        df = self._ranks_cmp.to_dataframe(untied=untied)
        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": RANKS_LABELS[untied]})
        return sns.heatmap(data=df, **kwargs)

    def corr(self, *, untied=False, **kwargs):
        corr = self._ranks_cmp.corr(untied=untied)
        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": "Correlation"})
        return sns.heatmap(data=corr, **kwargs)

    def cov(self, *, untied=False, **kwargs):
        cov = self._ranks_cmp.cov(untied=untied)
        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": "Covariance"})
        return sns.heatmap(data=cov, **kwargs)

    def r2_score(self, untied=False, **kwargs):
        r2 = self._ranks_cmp.r2_score(untied=untied)
        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": "$R^2$"})
        return sns.heatmap(data=r2, **kwargs)

    def distance(self, *, untied=False, metric="hamming", **kwargs):
        dis = self._ranks_cmp.distance(untied=untied, metric=metric)
        kwargs.setdefault("annot", True)
        kwargs.setdefault(
            "cbar_kws", {"label": f"{metric} distance".capitalize()}
        )
        return sns.heatmap(data=dis, **kwargs)

    def box(self, *, untied=False, **kwargs):
        df = self._ranks_cmp.to_dataframe(untied=untied)
        ax = sns.boxplot(data=df.T, **kwargs)

        ranks_label = RANKS_LABELS[untied]
        if kwargs.get("orient") in (None, "v"):
            ax.set_ylabel(ranks_label)
        else:
            ax.set_xlabel(ranks_label)

        return ax

    # DATAFRAME BASED  ========================================================
    # Thin wrapper around pandas.DataFrame.plot

    def bar(self, *, untied=False, **kwargs):
        df = self._ranks_cmp.to_dataframe(untied=untied)
        kwargs["ax"] = kwargs.get("ax") or plt.gca()
        ax = df.plot.bar(**kwargs)
        ax.set_ylabel(RANKS_LABELS[untied])
        return ax

    def barh(self, *, untied=False, **kwargs):
        df = self._ranks_cmp.to_dataframe(untied=untied)
        kwargs["ax"] = kwargs.get("ax") or plt.gca()
        ax = df.plot.barh(**kwargs)
        ax.set_xlabel(RANKS_LABELS[untied])
        return ax


# =============================================================================
# COMPARATOR
# =============================================================================


class RanksComparator(Mapping):
    def __init__(self, ranks):

        if isinstance(ranks, Mapping):
            ranks = ranks.copy()
        else:
            names = [r.method for r in ranks]
            ranks = dict(unique_names(names=names, elements=ranks))
            del names

        self._validate_ranks_with_same_atlernatives(ranks)
        self._ranks = ranks

    # INTERNALS ===============================================================
    def _validate_ranks_with_same_atlernatives(self, ranks):
        total, cnt = len(ranks), Counter()
        if total == 1:
            raise ValueError("Please provide more than one ranking")

        for rank in ranks.values():
            cnt.update(rank.alternatives)

        missing = {aname for aname, acnt in cnt.items() if acnt < total}
        if missing:
            miss_str = ", ".join(missing)
            raise ValueError(
                f"Some ranks miss the alternative/s: {miss_str!r}"
            )

        return missing

    # MAGIC! ==================================================================

    def __repr__(self):
        cls_name = type(self).__name__
        ranks_str = list(self._ranks.keys())
        return f"<{cls_name} ranks={ranks_str!r}>"

    def __getitem__(self, rname):
        return self._ranks[rname]

    def __iter__(self):
        return iter(self._ranks)

    def __len__(self):
        return len(self._ranks)

    def __hash__(self):
        return id(self)

    # TO DATA =================================================================

    def to_dataframe(self, *, untied=False):
        columns = {
            rank_name: rank.to_series(untied=untied)
            for rank_name, rank in self._ranks.items()
        }

        df = pd.DataFrame.from_dict(columns)
        df.columns.name = "Method"

        return df

    def corr(self, *, untied=False):
        return self.to_dataframe(untied=untied).corr()

    def cov(self, *, untied=False):
        return self.to_dataframe(untied=untied).cov()

    def r2_score(self, *, untied=False):
        df = self.to_dataframe(untied=untied)
        # here we are going to create a dict of dict
        rows = defaultdict(dict)

        # combine the methods pairwise
        for r0, r1 in it.combinations(df.columns, 2):
            r2_score = _skl_metrics.r2_score(df[r0], df[r1])

            # add the metrics in both directions
            rows[r0][r1] = r2_score
            rows[r1][r0] = r2_score

        # create the dataframe and change the nan for 1 (perfect R2)
        r2_df = pd.DataFrame.from_dict(rows).fillna(1)
        r2_df = r2_df[df.columns].loc[df.columns]

        r2_df.index.name = "Method"
        r2_df.columns.name = "Method"

        return r2_df

    def distance(self, *, untied=False, metric="hamming"):
        df = self.to_dataframe(untied=untied).T
        dis_array = distance.pdist(df, metric=metric)
        dis_mtx = distance.squareform(dis_array)
        dis_df = pd.DataFrame(
            dis_mtx, columns=df.index.copy(), index=df.index.copy()
        )
        return dis_df

    # ACCESSORS (YES, WE USE CACHED PROPERTIES IS THE EASIEST WAY) ============

    @property
    @functools.lru_cache(maxsize=None)
    def plot(self):
        """Plot accessor."""
        return RanksComparatorPlotter(self)
