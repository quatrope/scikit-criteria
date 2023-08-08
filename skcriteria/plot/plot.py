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

import pandas as pd

import seaborn as sns

from .objectives import Objective
from ..utils import AccessorABC


# =============================================================================
# PLOTTER OBJECT
# =============================================================================


class DecisionMatrixPlotter(AccessorABC):
    """DecisionMatrix plot utilities.

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
    - 'frontier': criteria pair-wise Pareto-Frontier.

    """

    _default_kind = "heatmap"

    def __init__(self, dm):
        self._dm = dm

    # PRIVATE =================================================================
    # This method are used "a lot" inside all the different plots, so we can
    # save some lines of code
    def _get_criteria_labels(self, **kwargs):
        kwargs.setdefault("fmt", "{criteria} {objective}")
        labels = self._dm._get_cow_headers(**kwargs)
        return pd.Series(labels, name="Criteria")

    @property
    def _ddf(self):
        # proxy to access the dataframe with the data
        ddf = self._dm.matrix
        ddf.columns = self._get_criteria_labels()
        return ddf

    @property
    def _wdf(self):
        # proxy to access the dataframe with the weights
        wdf = self._dm.weights.to_frame()
        wdf.index = self._get_criteria_labels()
        return wdf

    # HEATMAP =================================================================

    def _heatmap(self, df, **kwargs):
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
        ax = sns.boxplot(data=self._ddf, **kwargs)
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
        dm = self._dm
        import numpy as np

        dom = dm.dominance.dominance(strict=strict)
        bt = dm.dominance.bt().to_numpy().astype(str)
        eq = dm.dominance.eq().to_numpy().astype(str)

        annot = kwargs.pop("annot", True)
        if annot:
            annot = ""
            for elem in [r"$\succ", bt, "$/$=", eq, "$"]:
                annot = np.char.add(annot, elem)

        kwargs.setdefault("cbar", False)
        kwargs.setdefault("fmt", "")
        ax = self._heatmap(dom, annot=annot, **kwargs)

        return ax

    def frontier(
        self,
        x,
        y,
        *,
        strict=False,
        ax=None,
        legend=True,
        scatter_kws=None,
        line_kws=None,
    ):
        """Pareto frontier on two arbitrarily selected criteria.

        A selection of an alternative of an $A_o$ is a pareto-optimal solution
        when there is no other solution that selects an alternative that does
        not belong to $A_o$ such that it improves on one objective without
        worsening at least one of the others.

        From this point of view, the concept is used to analyze the possible
        optimal options of a solution given a variety of objectives or desires
        and one or more evaluation criteria.

        Given a "universe" of alternatives, one seeks to determine the set that
        are Pareto efficient (i.e., those alternatives that satisfy the
        condition of not being able to better satisfy one of those desires or
        objectives without worsening some other). That set of optimal
        alternatives establishes a "Pareto set" or the "Pareto Frontier".

        The study of the solutions in the frontier allows designers to analyze
        the possible alternatives within the established parameters, without
        having to analyze the totality of possible solutions.

        Parameters
        ----------
        x, y : str
            Criteria names.
            Variables that specify positions on the x and y axes.
        weighted: bool, default ``False``
            If its True the domination analysis is performed over the weighted
            matrix.
        strict: bool, default ``False``
            If True, strict dominance is evaluated.
        weighted: bool, default ``False``
            If True, the weighted matrix is evaluated.
        ax : :class:`matplotlib.axes.Axes`
            Pre-existing axes for the plot. Otherwise, call
            ``matplotlib.pyplot.gca`` internally.
        legend : bool, default ``True``
            If ``False``, no legend data is added and no legend is drawn.
        scatter_kws: dict, default ``None``
            Additional parameters passed to ``seaborn.scatterplot``.
        scatter_kws: dict, default ``None``
            Additional parameters passed to ``seaborn.lineplot``,
            except for ``estimator`` and ``sort``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        References
        ----------
        :cite:p:`enwiki:1107297090`
        :cite:p:`enwiki:1110412520`

        """
        # cut the dmatrix to only the necesary criteria
        sdm = self._dm[[x, y]]

        # extract the matrix
        df = sdm.matrix

        # draw the scatterplot ================================================
        scatter_kws = {} if scatter_kws is None else scatter_kws
        scatter_kws.setdefault("ax", ax)
        scatter_kws.setdefault("legend", legend)
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
            obj_x, obj_y = sdm.objectives

            non_dominated.iloc[0, 0] = (
                df[x].min() if obj_x is Objective.MAX else df[x].max()
            )
            non_dominated.iloc[2, 1] = (
                df[y].min() if obj_y is Objective.MAX else df[y].max()
            )

        # line style and frontier label
        frontier_ls, frontier_lb = (
            ("-", "Strict frontier") if strict else ("--", "Frontier")
        )

        # draw the line plot
        line_kws = {} if line_kws is None else line_kws
        line_kws.setdefault("alpha", 0.5)
        line_kws.setdefault("linestyle", frontier_ls)
        line_kws.setdefault("label", frontier_lb)
        line_kws.setdefault("legend", legend)

        sns.lineplot(
            x=x,
            y=y,
            data=non_dominated,
            estimator=None,
            sort=False,
            ax=ax,
            **line_kws,
        )

        # Set the labels
        xlabel, ylabel = self._get_criteria_labels(only=[x, y])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title="Alternatives")

        return ax
