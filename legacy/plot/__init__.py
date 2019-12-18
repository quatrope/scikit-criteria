#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# =============================================================================
# DOCS
# =============================================================================

"""Plotting utilities"""


__all__ = ["DataPlotMethods"]


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from matplotlib import cm

from ..validate import MIN, CRITERIA_STR
from .. import norm

from .radar import radar_plot
from .multihist import multihist_plot
from .scmtx import scmtx_plot
from .box import box_plot
from .violin import violin_plot
from .bars import bars_plot


# =============================================================================
# DECORATOR
# =============================================================================

_plot_types = set()


def _plot_type(method):
    _plot_types.add(method.__name__)
    return method


# =============================================================================
# CLASS
# =============================================================================

class DataPlotMethods(object):
    """Data plotting accessor and method

    Examples
    --------

    >>> data.plot()
    >>> data.plot.hist()
    >>> data.plot.scatter('x', 'y')
    >>> data.plot.radar()

    These plotting methods can also be accessed by calling the accessor as a
    method with the ``kind`` argument:
    ``data.plot(kind='violin')`` is equivalent to ``data.plot.violin()``

    """
    def __init__(self, data):
        self._data = data

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    def __call__(self, kind="radar", **kwargs):
        """Make plots of Data using matplotlib."""

        if kind not in _plot_types:
            msg = "Invalid kind '{}'. Chooce from: {}"
            raise ValueError(msg.format(kind, ", ".join(_plot_types)))
        method = getattr(self, kind)
        return method(**kwargs)

    def to_str(self):
        return "DataPlotMethods for {}".format(self._data)

    def preprocess(self, data, mnorm, wnorm,
                   anames, cnames, cmap,
                   weighted, show_criteria, min2max,
                   push_negatives, addepsto0):
        """Preprocess the data to be plotted.

        Parameters
        ----------
        data : skcritria.core.Data
            The data to be preprocessed.
        mnorm: string, callable
            Normalization method for the alternative matrix.
        wnorm : string, callable
            Normalization method for the weights array.
        anames : list of str or None
            The list of alternative names to be render in the plot.
            If is None then the alternative names of data are used.
        cnames : list of str or None
            The list of criteria names to be render in the plot.
            If is None then the criteria names of data are used.
        cmap : string or None
            Name of the color map to be used
            (https://matplotlib.org/users/colormaps.html)
        weighted : bool
            If the data must be weighted before redering.
        show_criteria : bool
            I the sense of optimality must be rendered in the plot.
        min2max : bool
            If true all the data of the minimization criteria are inverted
            before render.
        push_negatives : bool
            If True all the criterias with some value < 0 are incremented to
            be at least 0 in the minimun value.
        addepsto0 : bool
            If true add an small value to all the zeros inside the data.

        Returns
        -------

        preprocessed_data : dict
            All the data ready to be sended to a plot function

        """

        # extract all the data
        mtx = data.mtx
        criteria = data.criteria
        weights = data.weights
        anames = anames or data.anames
        cnames = cnames or data.cnames

        # push negatives
        if push_negatives:
            mtx = norm.push_negatives(mtx, axis=0)

        # prevent zeroes
        if addepsto0:
            mtx = norm.addepsto0(mtx, axis=0)

        # convert all minimun criteria to max
        if min2max:
            mincrits = np.squeeze(np.where(criteria == MIN))
            if np.any(mincrits):
                mincrits_inverted = 1.0 / mtx[:, mincrits]
                mtx = mtx.astype(mincrits_inverted.dtype.type)
                mtx[:, mincrits] = mincrits_inverted

        # normalization
        mtx = norm.norm(mnorm, mtx, criteria=criteria, axis=0)
        weights = (
            norm.norm(wnorm, weights, criteria=criteria)
            if weights is not None else None)

        # weight the data
        if weighted and weights is not None:
            mtx = np.multiply(mtx, weights)

        # labels for criteria
        criterias = (
            [" ({})".format(CRITERIA_STR[c]) for c in criteria]
            if show_criteria else
            [""] * len(criteria))

        if weights is not None:
            cnames = [
                "{}{}\n(w.{:.2f})".format(cn, cr, cw)
                for cn, cr, cw in zip(cnames, criterias, weights)]
        else:
            cnames = [
                "{}{}".format(cn, cr) for cn, cr in zip(cnames, criterias)]

        return {
            "mtx": mtx, "criteria": criteria, "weights": weights,
            "cmap": cm.get_cmap(name=cmap),
            "anames": anames, "cnames": cnames}

    def plot(self, func, mnorm="none", wnorm="none",
             anames=None, cnames=None, cmap=None,
             weighted=True, show_criteria=True, min2max=False,
             push_negatives=False, addepsto0=False, **kwargs):
        """Preprocess the data and send to the plot function *func*.

        Parameters
        ----------

        func : callable
            The function that make the plot. The return value of func
            are the return value of this method.
        mnorm: string, callable, optional (default="none")
            Normalization method for the alternative matrix.
        wnorm : string, callable, optional (default="none")
            Normalization method for the weights array.
        anames : list of str or None, optional (default=None)
            The list of alternative names to be render in the plot.
            If is None then the alternative names of data are used.
        cnames : list of str or None, optional (default=None)
            The list of criteria names to be render in the plot.
            If is None then the criteria names of data are used.
        cmap : string or None, optional (default=None)
            Name of the color map to be used
            (https://matplotlib.org/users/colormaps.html)
        weighted : bool, optional (default=True)
            If the data must be weighted before redering.
        show_criteria : bool, optional (default=True)
            I the sense of optimality must be rendered in the plot.
        min2max : bool, optional (default=False)
            If true all the data of the minimization criteria are inverted
            before render.
        push_negatives : bool, optional (default=False)
            If True all the criterias with some value < 0 are incremented to
            be at least 0 in the minimun value.
        addepsto0 : bool, optional (default=False)
            If true add an small value to all the zeros inside the data.
        kwargs :
            Arguments to send to *func*

        Returns
        -------
        The return value of *func*.

        Notes
        -----
        All the plot methods of Scikit-Criteria returns a matplotlib axis.

        """

        ppkwargs = self.preprocess(
            data=self._data, mnorm=mnorm, wnorm=wnorm,
            anames=anames, cnames=cnames, cmap=cmap,
            weighted=weighted, show_criteria=show_criteria,
            addepsto0=addepsto0, min2max=min2max,
            push_negatives=push_negatives)
        kwargs.update(ppkwargs)
        return func(**kwargs)

    @_plot_type
    def radar(self, **kwargs):
        """Creates a radar chart, also known as a spider or star chart
        (http://en.wikipedia.org/wiki/Radar_chart).

        A radar chart is a graphical method of displaying multivariate data in
        the form of a two-dimensional chart of three or more quantitative
        variables represented on axes starting from the same point. The
        relative position and angle of the axes is typically uninformative.

        Parameters
        ----------
        frame : {"polygon", "circle"}
            Shape of frame surrounding axes.
        ax : None or PolarAxes, optional (default=None)
            Axis where the radar must be redered. Is is None a new axis are
            created.
        legendcol : int, optional (default=5)
            How many columns must has the legend.
        subplots_kwargs : dict or None, optional (default=None)
            Argument to send to ``matplotlib.pyplot.subplots`` if axis is None.
            If axis is not None, subplots_kwargs are ignored.

        Returns
        -------
        ax : matplotlib.projections.polar.PolarAxes
            Axis where the radar are rendered

        See Also
        --------
        DataPlotMethods.plot : To check all the available parameters


        Notes
        -----
        All the parameters in ``plot()`` are supported; but by default
        this method override some default values:

        - ``show_criteria=False``
        - ``min2max=True``
        - ``push_negatives=True``
        - ``addepsto0=True``

        """
        kwargs.setdefault("show_criteria", False)
        kwargs.setdefault("min2max", True)
        kwargs.setdefault("push_negatives", True)
        kwargs.setdefault("addepsto0", True)
        return self.plot(radar_plot, **kwargs)

    @_plot_type
    def hist(self, **kwargs):
        return self.plot(multihist_plot, **kwargs)

    @_plot_type
    def scatter(self, **kwargs):
        return self.plot(scmtx_plot, **kwargs)

    @_plot_type
    def box(self, **kwargs):
        return self.plot(box_plot, **kwargs)

    @_plot_type
    def violin(self, **kwargs):
        return self.plot(violin_plot, **kwargs)

    @_plot_type
    def bars(self, **kwargs):
        self.plot(bars_plot, **kwargs)
