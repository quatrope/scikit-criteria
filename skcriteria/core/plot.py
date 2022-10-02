#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the DecisionMatrix object."""

# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from .objectives import Objective
from ..utils import AccessorABC


# =============================================================================
# PLOTTER OBJECT
# =============================================================================


class DecisionMatrixPlotter(AccessorABC):
    """Make plots of DecisionMatrix.

    Kind of plot to produce:

    - 'heatmap' : criteria heat-map (default).
    - 'wheatmap' : weights heat-map.
    - 'bar' : criteria vertical bar plot.
    - 'wbar' : weights vertical bar plot.
    - 'barh' : criteria horizontal bar plot.
    - 'wbarh' : weights horizontal bar plot.
    - 'hist' : criteria histogram.
    - 'whist' : weights histogram.
    - 'box' : criteria boxplot.
    - 'wbox' : weights boxplot.
    - 'kde' : criteria Kernel Density Estimation plot.
    - 'wkde' : weights Kernel Density Estimation plot.
    - 'ogive' : criteria empirical cumulative distribution plot.
    - 'wogive' : weights empirical cumulative distribution plot.
    - 'area' : criteria area plot.
    - 'dominance': the dominance matrix as a heatmap.

    """

    _default_kind = "heatmap"

    def __init__(self, dm):
        self._dm = dm

    # PRIVATE =================================================================
    # This method are used "a lot" inside all the different plots, so we can
    # save some lines of code

    @property
    def _ddf(self):
        # proxy to access the dataframe with the data
        return self._dm.matrix

    @property
    def _wdf(self):
        # proxy to access the dataframe with the weights
        return self._dm.weights.to_frame()

    def _get_criteria_labels(self, **kwargs):
        kwargs.setdefault("fmt", "{criteria} {objective}")
        return self._dm._get_cow_headers(**kwargs)

    # HEATMAP =================================================================

    def _heatmap(self, df, **kwargs):
        kwargs.setdefault("cmap", plt.cm.get_cmap())
        ax = sns.heatmap(df, **kwargs)
        return ax

    def heatmap(self, **kwargs):
        """Plot the alternative matrix as a color-encoded matrix.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        kwargs.setdefault("annot", True)
        ax = self._heatmap(self._ddf, **kwargs)

        xticklabels = self._get_criteria_labels()
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Criteria")
        ax.set_ylabel("Alternatives")

        return ax

    def wheatmap(self, **kwargs):
        """Plot weights as a color-encoded matrix.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        kwargs.setdefault("annot", True)
        ax = self._heatmap(self._wdf.T, **kwargs)

        xticklabels = self._get_criteria_labels()
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Criteria")

        if "ax" not in kwargs:
            # if the ax is provided by the user we assume that the figure
            # is already with the expected size. If it's not, we resize the
            # height to 1/5 of the original size.
            fig = ax.get_figure()
            size = fig.get_size_inches() / [1, 5]
            fig.set_size_inches(size)

        return ax

    # BAR =====================================================================

    def bar(self, **kwargs):
        """Criteria vertical bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.bar``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._ddf.plot.bar(**kwargs)
        ax.set_xlabel("Alternatives")
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    def wbar(self, **kwargs):
        """Weights vertical bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.bar``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._wdf.T.plot.bar(**kwargs)
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    # BARH ====================================================================

    def barh(self, **kwargs):
        """Criteria horizontal bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.barh``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._ddf.plot.barh(**kwargs)
        ax.set_ylabel("Alternatives")
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    def wbarh(self, **kwargs):
        """Weights horizontal bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.barh``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._wdf.T.plot.barh(**kwargs)
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    # HIST ====================================================================

    def hist(self, **kwargs):
        """Draw one histogram of the criteria.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.histplot(self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._get_criteria_labels())
        return ax

    def whist(self, **kwargs):
        """Draw one histogram of the weights.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.histplot(self._wdf.T, **kwargs)
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    # BOX =====================================================================

    def box(self, **kwargs):
        """Make a box plot of the criteria.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        orient = kwargs.setdefault("orient", "v")

        ax = sns.boxplot(data=self._ddf, **kwargs)

        if orient == "v":
            xticklabels = self._get_criteria_labels()
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("Criteria")
        elif orient == "h":
            yticklabels = self._get_criteria_labels()
            ax.set_yticklabels(yticklabels)
            ax.set_ylabel("Criteria")

        return ax

    def wbox(self, **kwargs):
        """Make a box plot of the weights.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.boxplot(data=self._wdf, **kwargs)
        return ax

    # KDE =====================================================================

    def kde(self, **kwargs):
        """Criteria kernel density plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.kdeplot(data=self._ddf, **kwargs)
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    def wkde(self, **kwargs):
        """Weights kernel density plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.kdeplot(data=self._wdf, **kwargs)
        return ax

    # OGIVE ===================================================================

    def ogive(self, **kwargs):
        """Criteria empirical cumulative distribution plot.

        In statistics, an empirical distribution function (eCDF) is the
        distribution function associated with the empirical measure of a
        sample. This cumulative distribution function is a step function that
        jumps up by 1/n at each of the n data points. Its value at any
        specified value of the measured variable is the fraction of
        observations of the measured variable that are less than or equal to
        the specified value.

        .. _empirical distribution function:
            https://en.wikipedia.org/wiki/Empirical_distribution_function

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.ecdfplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.ecdfplot(data=self._ddf, **kwargs)
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    def wogive(self, **kwargs):
        """Weights empirical cumulative distribution plot.

        In statistics, an empirical distribution function (eCDF) is the
        distribution function associated with the empirical measure of a
        sample. This cumulative distribution function is a step function that
        jumps up by 1/n at each of the n data points. Its value at any
        specified value of the measured variable is the fraction of
        observations of the measured variable that are less than or equal to
        the specified value.

        .. _empirical distribution function:
            https://en.wikipedia.org/wiki/Empirical_distribution_function

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.ecdfplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.ecdfplot(data=self._wdf, **kwargs)
        return ax

    # AREA ====================================================================

    def area(self, **kwargs):
        """Draw an criteria stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the matplotlib area function.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed and are documented in
            :meth:`DataFrame.plot.area`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Area plot, or array of area plots if subplots is True.

        """
        ax = self._ddf.plot.area(**kwargs)
        ax.set_xlabel("Alternatives")
        if kwargs.get("legend", True):
            legend = self._get_criteria_labels()
            ax.legend(legend)
        return ax

    # DOMINANCE ===============================================================

    def dominance(self, *, strict=False, **kwargs):
        """Plot dominance as a color-encoded matrix.

        In order to evaluate the dominance of an alternative *a0* over an
        alternative *a1*, the algorithm evaluates that *a0* is better in at
        least one criterion and that *a1* is not better in any criterion than
        *a0*. In the case that ``strict = True`` it also evaluates that there
        are no equal criteria.

        Parameters
        ----------
        strict: bool, default ``False``
            If True, strict dominance is evaluated.
        **kwargs:
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        dom = self._dm.dominance.dominance(strict=strict)

        kwargs.setdefault("cbar", False)
        ax = self._heatmap(dom, **kwargs)

        ax.set_title("Strict dominance" if strict else "Dominance")
        ax.set_ylabel("Alternatives")
        ax.set_xlabel("Alternatives")

        return ax

    # FRONTIER

    def frontier(
        self, x, y, *, strict=False, ax=None, scatter_kws=None, line_kws=None
    ):

        # cut the dmatrix to only the necesary criteria
        sdm = self._dm[[x, y]]

        # extract objectives and matrix
        df, (obj_x, obj_y) = sdm.matrix, sdm.objectives

        # draw the scatterplot ================================================
        scatter_kws = {} if scatter_kws is None else scatter_kws
        scatter_kws.setdefault("ax", ax)
        ax = sns.scatterplot(x=x, y=y, data=df, hue=df.index, **scatter_kws)

        # draw the frontier ===================================================
        # Get the non dominated alternatives.
        # This alternatives create the frontier
        non_dominated = df[
            ~sdm.dominance.dominated(strict=strict)
        ].sort_values([x, y])

        # if we only have one alternative in the frontier but we have more
        # alternatives we draw a limit around all the dominated one.
        if len(non_dominated) == 1 and len(sdm.alternatives) > 1:
            non_dominated = pd.concat([non_dominated] * 3, ignore_index=True)

            # esto cambia si x o y son a minimizar
            non_dominated.iloc[0, 0] = (
                df[x].min() if obj_x is Objective.MAX else df[x].max()
            )
            non_dominated.iloc[2, 1] = (
                df[y].min() if obj_y is Objective.MAX else df[y].max()
            )

        # draw the line plot
        line_kws = {} if line_kws is None else line_kws
        line_kws.setdefault("alpha", 0.5)
        line_kws.setdefault("linestyle", "--")
        line_kws.setdefault("label", "Frontier")

        sns.lineplot(
            x=x,
            y=y,
            data=non_dominated,
            estimator=None,
            sort=False,
            ax=ax,
            **line_kws,
        )

        # Set the title
        title = "Strict frontier" if strict else "Frontier"
        xlabel, ylabel = self._get_criteria_labels(only=[x, y])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax
