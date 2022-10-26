#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Ranking comparison routines."""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import itertools as it
from collections import defaultdict

import matplotlib.pyplot as plt

import pandas as pd

from scipy.spatial import distance

import seaborn as sns

from sklearn import metrics as _skl_metrics

from ..core import SKCMethodABC
from ..madm import RankResult
from ..utils import AccessorABC, Bunch, unique_names


# =============================================================================
# CONSTANTS
# =============================================================================

RANKS_LABELS = {
    True: "Untied ranks (lower is better)",
    False: "Ranks (lower is better)",
}


# =============================================================================
# COMPARATOR
# =============================================================================


class RanksComparator(SKCMethodABC):
    """Rankings comparator object.

    This class is intended to contain a collection of rankings on which you
    want to do comparative analysis.

    All rankings must have exactly the same alternatives, although their order
    may vary.

    All methods support the ``untied`` parameter, which serves to untie
    rankings in case there are results that can assign more than one
    alternative to the same position (e.g.``ELECTRE2``).

    Parameters
    ----------
    ranks : list
        List of (name, ranking) tuples of ``skcriteria.madm.RankResult``
        with the same alternatives.

    See Also
    --------
    skcriteria.cmp.mkrank_cmp : Convenience function for simplified
        ranks comparator construction.

    """

    _skcriteria_dm_type = "ranks_comparator"
    _skcriteria_parameters = ["ranks"]

    def __init__(self, ranks):
        ranks = list(ranks)
        self._validate_ranks(ranks)
        self._ranks = ranks

    # INTERNALS ===============================================================
    def _validate_ranks(self, ranks):

        if len(ranks) <= 1:
            raise ValueError("Please provide more than one ranking")

        used_names = set()
        first_alternatives = set(ranks[0][1].alternatives)
        for name, part in ranks:

            if not isinstance(name, str):
                raise ValueError("'name' must be instance of str")

            if not isinstance(part, RankResult):
                raise TypeError("ranks must be instance of madm.RankResult")

            if name in used_names:
                raise ValueError(f"Duplicated name {name!r}")
            used_names.add(name)

            diff = first_alternatives.symmetric_difference(part.alternatives)
            if diff:
                miss_str = ", ".join(diff)
                raise ValueError(
                    f"Some ranks miss the alternative/s: {miss_str!r}"
                )

    # PROPERTIES ==============================================================
    @property
    def ranks(self):
        """List of ranks in the comparator."""
        return list(self._ranks)

    @property
    def named_ranks(self):
        """Dictionary-like object, with the following attributes.

        Read-only attribute to access any rank parameter by user given name.
        Keys are ranks names and values are rannks parameters.

        """
        return Bunch("ranks", dict(self.ranks))

    # MAGIC! ==================================================================

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        cls_name = type(self).__name__
        ranks_names = [rn for rn, _ in self._ranks]
        return f"<{cls_name} [ranks={ranks_names!r}]>"

    def __len__(self):
        """Return the number of rankings to compare."""
        return len(self._ranks)

    def __getitem__(self, ind):
        """Return a sub-comparator or a single ranking in the pipeline.

        Indexing with an integer will return an ranking; using a slice
        returns another RankComparator instance which copies a slice of this
        RankComparator. This copy is shallow: modifying ranks in the
        sub-comparator will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.

        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                cname = type(self).__name__
                raise ValueError(f"{cname} slicing only supports a step of 1")
            return self.__class__(self.ranks[ind])
        elif isinstance(ind, int):
            return self._ranks[ind][-1]
        elif isinstance(ind, str):
            return self.named_ranks[ind]
        raise KeyError(ind)

    def __hash__(self):
        """x.__hash__() <==> hash(x)."""
        return id(self)

    # TO DATA =================================================================

    def to_dataframe(self, *, untied=False):
        """Convert the entire RanksComparator into a dataframe.

        The alternatives are the rows, and the different rankings are the
        columns.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.

        Returns
        -------
        :py:class:`pd.DataFrame`
            A RanksComparator as pandas DataFrame.

        """
        columns = {
            rank_name: rank.to_series(untied=untied)
            for rank_name, rank in self._ranks
        }

        df = pd.DataFrame.from_dict(columns)
        df.columns.name = "Method"

        return df

    def corr(self, *, untied=False, **kwargs):
        """Compute pairwise correlation of rankings, excluding NA/null values.

        By default the pearson correlation coefficient is used.

        Please check the full documentation of a ``pandas.DataFrame.corr()``
        method for details about the implementation.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the
            ``pandas.DataFrame.corr()`` method.

        Returns
        -------
        :py:class:`pd.DataFrame`
            A DataFrame with the correlation between rankings.

        """
        return self.to_dataframe(untied=untied).corr(**kwargs)

    def cov(self, *, untied=False, **kwargs):
        """Compute pairwise covariance of rankings, excluding NA/null values.

        Please check the full documentation of a ``pandas.DataFrame.cov()``
        method for details about the implementation.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the
            ``pandas.DataFrame.cov()`` method.

        Returns
        -------
        :py:class:`pd.DataFrame`
            A DataFrame with the covariance between rankings.

        """
        return self.to_dataframe(untied=untied).cov(**kwargs)

    def r2_score(self, *, untied=False, **kwargs):
        """Compute pairwise coefficient of determination regression score \
        function of rankings, excluding NA/null values.

        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse).

        Please check the full documentation of a ``sklearn.metrics.r2_score``
        function for details about the implementation and the behaviour.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the
            ``sklearn.metrics.r2_score()`` function.

        Returns
        -------
        :py:class:`pd.DataFrame`
            A DataFrame with the coefficient of determination between rankings.

        """
        df = self.to_dataframe(untied=untied)
        # here we are going to create a dict of dict
        rows = defaultdict(dict)

        # combine the methods pairwise
        for r0, r1 in it.combinations(df.columns, 2):
            r2_score = _skl_metrics.r2_score(df[r0], df[r1], **kwargs)

            # add the metrics in both directions
            rows[r0][r1] = r2_score
            rows[r1][r0] = r2_score

        # create the dataframe and change the nan for 1 (perfect R2)
        r2_df = pd.DataFrame.from_dict(rows).fillna(1)
        r2_df = r2_df[df.columns].loc[df.columns]

        r2_df.index.name = "Method"
        r2_df.columns.name = "Method"

        return r2_df

    def distance(self, *, untied=False, metric="hamming", **kwargs):
        """Compute pairwise distance between rankings.

        By default the 'hamming' distance is used, which is simply the
        proportion of disagreeing components in Two rankings.

        Please check the full documentation of a
        ``scipy.spatial.distance.pdist`` function for details about the
        implementation and the behaviour.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        metric: str or function, default ``"hamming"``
            The distance metric to use. The distance function can
            be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
            'jaccard', 'jensenshannon', 'kulczynski1',
            'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            'sqeuclidean', 'yule'.
        kwargs:
            Other keyword arguments are passed to the
            ``scipy.spatial.distance.pdist()`` function.

        Returns
        -------
        :py:class:`pd.DataFrame`
            A DataFrame with the distance between rankings.

        """
        df = self.to_dataframe(untied=untied).T
        dis_array = distance.pdist(df, metric=metric, **kwargs)
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


# =============================================================================
# PLOTTER
# =============================================================================


class RanksComparatorPlotter(AccessorABC):
    """RanksComparator plot utilities.

    Kind of plot to produce:

    - 'flow' : Changes in the rankings of the alternatives as flow lines
      (default)
    - 'reg' : Pairwise rankings data and a linear regression model fit plot.
    - 'heatmap' : Rankings as a color-encoded matrix.
    - 'corr' : Pairwise correlation of rankings as a color-encoded matrix.
    - 'cov' : Pairwise covariance of rankings as a color-encoded matrix.
    - 'r2_score' : Pairwise coefficient of determination regression score \
      function of rankings as a color-encoded matrix.
    - 'distance' : Pairwise distance between rankings as a color-encoded \
      matrix.
    - 'box' : Box-plot of rankings with respect to alternatives
    - 'bar' : Ranking of alternatives by method with vertical bars.
    - 'barh' : Ranking of alternatives by method with horizontal bars.

    """

    _default_kind = "flow"

    def __init__(self, ranks_cmp):
        self._ranks_cmp = ranks_cmp

    # MANUAL MADE PLOT ========================================================
    # These plots have a much more manually orchestrated code.

    def flow(self, *, untied=False, grid_kws=None, **kwargs):
        """Represents changes in the rankings of the alternatives as lines \
        flowing through the ranking-methods.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        grid_kws: dict or None
            Dict with keyword arguments passed to
            ``matplotlib.axes.plt.Axes.grid``
        kwargs:
            Other keyword arguments are passed to the ``seaborn.lineplot()``
            function. except for data, estimator and sort.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
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
        r2_kws=None,
        **kwargs,
    ):
        """Plot a pairwise rankings data and a linear regression model fit.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        r2 : bool, default ``True``
            If True, the coefficient of determination results are added to the
            regression legend.
        palette:  matplotlib/seaborn color palette, default ``None``
            Set of colors for mapping the hue variable.
        legend: bool, default ``True``
            If False, suppress the legend for semantic variables.
        r2_fmt: str, default ``"2.g"``
            String formatting code to use when adding the coefficient of
            determination.
        r2_kws: dict or None
            Dict with keywords arguments passed to
            ``sklearn.metrics.r2_score()`` function.
        kwargs:
            Other keyword arguments are passed to the ``seaborn.lineplot()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        df = self._ranks_cmp.to_dataframe(untied=untied)

        # Just to ensure that no manual color reaches regplot
        if "color" in kwargs:
            cls_name = type(self).__name__
            raise TypeError(
                f"{cls_name}.reg() got an unexpected keyword argument 'color'"
            )

        # if there is a custom axis, we take it out
        ax = kwargs.pop("ax", None)

        # r2
        if legend and r2:
            r2_kws = {} if r2_kws is None else r2_kws
            r2_df = self._ranks_cmp.r2_score(untied=untied, **r2_kws)

        # we create the infinite cycle of colors for the palette,
        # so we take out as we need
        colors = it.cycle(sns.color_palette(palette=palette))

        # pairwise ranks iteration
        for x, y in it.combinations(df.columns, 2):
            color = next(colors)

            # The r2 correlation index
            r2_label = ""
            if legend and r2:
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
        """Plot the rankings as a color-encoded matrix.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the ``seaborn.heatmap()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        df = self._ranks_cmp.to_dataframe(untied=untied)
        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": RANKS_LABELS[untied]})
        return sns.heatmap(data=df, **kwargs)

    def corr(self, *, untied=False, corr_kws=None, **kwargs):
        """Plot the pairwise correlation of rankings as a color-encoded matrix.

        By default the pearson correlation coefficient is used.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        corr_kws: dict or None
            Dict with keywords arguments passed the
            ``pandas.DataFrame.corr()`` method.
        kwargs:
            Other keyword arguments are passed to the ``seaborn.heatmap()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        corr_kws = {} if corr_kws is None else corr_kws
        corr = self._ranks_cmp.corr(untied=untied, **corr_kws)

        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": "Correlation"})
        return sns.heatmap(data=corr, **kwargs)

    def cov(self, *, untied=False, cov_kws=None, **kwargs):
        """Plot the pairwise covariance of rankings as a color-encoded matrix.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        cov_kws: dict or None
            Dict with keywords arguments passed the
            ``pandas.DataFrame.cov()`` method.
        kwargs:
            Other keyword arguments are passed to the ``seaborn.heatmap()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        cov_kws = {} if cov_kws is None else cov_kws
        cov = self._ranks_cmp.cov(untied=untied, **cov_kws)

        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": "Covariance"})
        return sns.heatmap(data=cov, **kwargs)

    def r2_score(self, untied=False, r2_kws=None, **kwargs):
        """Plot the pairwise coefficient of determination regression score \
        function of rankings as a color-encoded matrix.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        cov_kws: dict or None
            Dict with keywords arguments passed the
            ``pandas.DataFrame.cov()`` method.
        kwargs:
            Other keyword arguments are passed to the ``seaborn.heatmap()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        r2_kws = {} if r2_kws is None else r2_kws
        r2 = self._ranks_cmp.r2_score(untied=untied, **r2_kws)

        kwargs.setdefault("annot", True)
        kwargs.setdefault("cbar_kws", {"label": "$R^2$"})
        return sns.heatmap(data=r2, **kwargs)

    def distance(
        self, *, untied=False, metric="hamming", distance_kws=None, **kwargs
    ):
        """Plot the pairwise distance between rankings as a color-encoded \
        matrix.

        By default the 'hamming' distance is used, which is simply the
        proportion of disagreeing components in Two rankings.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        metric: str or function, default ``"hamming"``
            The distance metric to use. The distance function can
            be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
            'jaccard', 'jensenshannon', 'kulczynski1',
            'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            'sqeuclidean', 'yule'.
        distance_kws: dict or None
            Dict with keywords arguments passed the
            ``scipy.spatial.distance.pdist`` function
        kwargs:
            Other keyword arguments are passed to the ``seaborn.heatmap()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        distance_kws = {} if distance_kws is None else distance_kws
        dis = self._ranks_cmp.distance(
            untied=untied, metric=metric, **distance_kws
        )

        kwargs.setdefault("annot", True)
        kwargs.setdefault(
            "cbar_kws", {"label": f"{metric} distance".capitalize()}
        )
        return sns.heatmap(data=dis, **kwargs)

    def box(self, *, untied=False, **kwargs):
        """Draw a boxplot to show rankings with respect to alternatives.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the ``seaborn.boxplot()``
            function.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
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
        """Draw plot that presents ranking of alternatives by method with \
        vertical bars.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the
            ``pandas.Dataframe.plot.bar()`` method.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        df = self._ranks_cmp.to_dataframe(untied=untied)
        kwargs["ax"] = kwargs.get("ax") or plt.gca()
        ax = df.plot.bar(**kwargs)
        ax.set_ylabel(RANKS_LABELS[untied])
        return ax

    def barh(self, *, untied=False, **kwargs):
        """Draw plot that presents ranking of alternatives by method with \
        horizontal bars.

        Parameters
        ----------
        untied: bool, default ``False``
            If it is ``True`` and any ranking has ties, the
            ``RankResult.untied_rank_`` property is used to assign each
            alternative a single ranked order. On the other hand, if it is
            ``False`` the rankings are used as they are.
        kwargs:
            Other keyword arguments are passed to the
            ``pandas.Dataframe.plot.barh()`` method.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        df = self._ranks_cmp.to_dataframe(untied=untied)
        kwargs["ax"] = kwargs.get("ax") or plt.gca()
        ax = df.plot.barh(**kwargs)
        ax.set_xlabel(RANKS_LABELS[untied])
        return ax


# =============================================================================
# FACTORY
# =============================================================================


def mkrank_cmp(*ranks):
    """Construct a RankComparator from the given rankings.

    This is a shorthand for the RankComparator constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the method attribute of the rankings automatically.

    Parameters
    ----------
    *ranks: list of RankResult objects
        List of the scikit-criteria RankResult objcects.

    Returns
    -------
    rcmp : RanksComparator
        Returns a scikit-criteria :class:`RanksComparator` object.

    """
    names = [r.method for r in ranks]
    named_ranks = unique_names(names=names, elements=ranks)
    return RanksComparator(named_ranks)
