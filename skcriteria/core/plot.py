#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the DecisionMatrix object."""

# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt

import seaborn as sns


# =============================================================================
# PLOTTER OBJECT
# =============================================================================
class DecisionMatrixPlotter:
    """Make plots of DecisionMatrix."""

    def __init__(self, dm):
        self._dm = dm

    # INTERNAL ================================================================

    def __call__(self, plot_kind="heatmap", **kwargs):
        """Make plots of a decision matrix.

        Parameters
        ----------
        plot_kind : str
            The kind of plot to produce:
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
                - 'wkde' : weightsKernel Density Estimation plot.
                - 'area' : criteria area plot.

        **kwargs
            Options to pass to subjacent plotting method.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them
           The ax used by the plot

        """
        if plot_kind.startswith("_"):
            raise ValueError(f"invalid plot_kind name '{plot_kind}'")
        method = getattr(self, plot_kind, None)
        if not callable(method):
            raise ValueError(f"invalid plot_kind name '{plot_kind}'")
        return method(**kwargs)

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

    @property
    def _criteria_labels(self):
        # list with all the criteria + objectives
        dm = self._dm
        labels = [
            f"{c} {o.to_string()}" for c, o in zip(dm.criteria, dm.objectives)
        ]
        return labels

    # HEATMAP =================================================================

    def _heatmap(self, df, **kwargs):
        kwargs.setdefault("annot", True)
        kwargs.setdefault("cmap", plt.cm.get_cmap())
        ax = sns.heatmap(df, **kwargs)
        return ax

    def heatmap(self, **kwargs):
        """Plot the alternative matrix as a color-encoded matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._heatmap(self._ddf, **kwargs)
        ax.set_xticklabels(self._criteria_labels)
        ax.set_ylabel("Alternatives")
        ax.set_xlabel("Criteria")
        return ax

    def wheatmap(self, **kwargs):
        """Plot weights as a color-encoded matrix.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.heatmap``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._heatmap(self._wdf.T, **kwargs)
        ax.set_xticklabels(self._criteria_labels)
        ax.set_xlabel("Criteria")

        if "ax" not in kwargs:
            # if the ax is provided by the user we assume that the figure
            # is already setted to the expected size. If it's not we resize the
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.bar``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._ddf.plot.bar(**kwargs)
        ax.set_xlabel("Alternatives")
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.bar``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._wdf.T.plot.bar(**kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.barh``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._ddf.plot.barh(**kwargs)
        ax.set_ylabel("Alternatives")
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``DataFrame.plot.barh``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = self._wdf.T.plot.barh(**kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
        return ax

    # HIST ====================================================================

    def hist(self, **kwargs):
        """Draw one histogram of the criteria.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.histplot(self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
        return ax

    def whist(self, **kwargs):
        """Draw one histogram of the weights.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.histplot(self._wdf.T, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        orient = kwargs.setdefault("orient", "v")

        ax = sns.boxplot(data=self._ddf, **kwargs)

        if orient == "v":
            ax.set_xticklabels(self._criteria_labels)
            ax.set_xlabel("Criteria")
        elif orient == "h":
            ax.set_yticklabels(self._criteria_labels)
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
        **kwargs
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.kdeplot(data=self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
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
        **kwargs
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
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.ecdfplot``.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        """
        ax = sns.ecdfplot(data=self._ddf, **kwargs)
        if kwargs.get("legend", True):
            ax.legend(self._criteria_labels)
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
        **kwargs
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
        **kwargs
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
            ax.legend(self._criteria_labels)
        return ax
