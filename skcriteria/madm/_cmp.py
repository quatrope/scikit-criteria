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

from collections import Counter
from collections.abc import Mapping
import itertools as it


import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn import metrics as _skl_metrics

from ..utils import unique_names

# =============================================================================
# CLASSES
# =============================================================================

_INSTANCE_ATTRIBUTE = "_singleton_instance"


class Singleton:
    def __new__(cls, *args, **kwargs):
        if _INSTANCE_ATTRIBUTE not in vars(cls):
            instance = super().__new__(cls, *args, **kwargs)
            setattr(cls, _INSTANCE_ATTRIBUTE, instance)
        return getattr(cls, _INSTANCE_ATTRIBUTE)


class _RanksComparator(Singleton):

    # INTERNALS ===============================================================
    def __repr__(self):
        return type(self).__name__

    def _validate_ranks_with_same_atlernatives(self, ranks):
        total, cnt = len(ranks), Counter()
        if total == 1:
            raise ValueError("Please provide more than one ranking")

        for rank in ranks.values():
            cnt.update(rank.alternatives)

        missing = {aname for aname, acnt in cnt.items() if acnt < total}
        if missing:
            miss_str = f", ".join(missing)
            raise ValueError(
                f"Some ranks miss the alternative/s: {miss_str!r}"
            )

        return missing

    # TO DATA =================================================================
    def dataframe_merge(
        self,
        ranks,
        *,
        untied=False,
    ):

        if not isinstance(ranks, Mapping):
            names = [r.method for r in ranks]
            ranks = dict(unique_names(names=names, elements=ranks))
            del names

        self._validate_ranks_with_same_atlernatives(ranks)

        columns = {
            rank_name: rank.to_series(untied=untied)
            for rank_name, rank in ranks.items()
        }

        df = pd.DataFrame.from_dict(columns)
        df.columns.name = "Method"

        return df

    # MANUAL MADE PLOT ========================================================
    # These plots have a much more manually orchestrated code.

    def flow(self, ranks, *, untied=False, grid_kws=None, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)

        ax = sns.lineplot(data=df.T, estimator=None, sort=False, **kwargs)

        grid_kws = {} if grid_kws is None else grid_kws
        grid_kws.setdefault("alpha", 0.3)
        ax.grid(**grid_kws)

        ax.set_ylabel(
            "{untied}Ranks".format(
                untied="Untied " if untied else ""
            ).capitalize()
        )
        ax.invert_yaxis()

        return ax

    def reg(
        self,
        ranks,
        *,
        untied=False,
        r2=True,
        palette=None,
        legend=True,
        **kwargs,
    ):

        df = self.dataframe_merge(ranks, untied=untied)

        # Just to ensure that no manual color reaches regalot
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

        # combinamos los rankings de dos en dos
        for x, y in it.combinations(df.columns, 2):
            color = next(colors)

            # The r2 correlation index
            r2_label = ""
            if r2:
                r2_score = _skl_metrics.r2_score(df[x], df[y])
                r2_label = f" - R2={r2_score:.4f}"

            label = "x={x}, y={y}{r2}".format(x=x, y=y, r2=r2_label)
            ax = sns.regplot(
                x=x, y=y, data=df, ax=ax, label=label, color=color, **kwargs
            )

        ax.set(xlabel="x", ylabel="y")

        if legend:
            ax.legend()

        return ax

    # SEABORN BASED ===========================================================
    # Thin wrapper around seaborn plots

    def heatmap(self, ranks, *, untied=False, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)
        kwargs.setdefault("annot", True)
        return sns.heatmap(data=df, **kwargs)

    def corr(self, ranks, *, untied=False, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)
        kwargs.setdefault("annot", True)
        return sns.heatmap(data=df.corr(), **kwargs)

    def cov(self, ranks, *, untied=False, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)
        kwargs.setdefault("annot", True)
        return sns.heatmap(data=df.cov(), **kwargs)

    def box(self, ranks, *, untied=False, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)
        ax = sns.boxplot(data=df.T, **kwargs)
        ax.set_ylabel("Ranks")
        return ax

    # DATAFRAME BASED  ========================================================
    # Thin wrapper around pandas.DataFrame.plot

    def bar(self, ranks, *, untied=False, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)
        kwargs["ax"] = kwargs.get("ax") or plt.gca()
        ax = df.plot.bar(**kwargs)
        ax.set_ylabel("Ranks")
        return ax

    def barh(self, ranks, *, untied=False, **kwargs):
        df = self.dataframe_merge(ranks, untied=untied)
        kwargs["ax"] = kwargs.get("ax") or plt.gca()
        ax = df.plot.barh(**kwargs)
        ax.set_xlabel("Ranks")
        return ax


# =============================================================================
# The instance!
# =============================================================================

#: Unique instance of the _RanksComparator
ranks_cmp = _RanksComparator()
