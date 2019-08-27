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
# META
# =============================================================================

""" Create a scatter-plot matrix using Matplotlib. """

__author__ = "adrn <adrn@astro.columbia.edu>"


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import matplotlib.pyplot as plt


# =============================================================================
# FUNCTIONS
# =============================================================================

def scatter_plot_matrix(
        data, labels, colors, axes=None,
        subplots_kwargs=None, scatter_kwargs=None, hist_kwargs=None):
    """Create a scatter plot matrix from the given data.

        Parameters
        ----------
        data : numpy.ndarray
            A numpy array containined the scatter data to plot. The data
            should be shape MxN where M is the number of dimensions and
            with N data points.
        labels : numpy.ndarray (optional)
            A numpy array of length M containing the axis labels.
        axes : matplotlib Axes array (optional)
            If you've already created the axes objects, pass this in to
            plot the data on that.
        subplots_kwargs : dict (optional)
            A dictionary of keyword arguments to pass to the
            matplotlib.pyplot.subplots call. Note: only relevant if axes=None.
        scatter_kwargs : dict (optional)
            A dictionary of keyword arguments to pass to the
            matplotlib.pyplot.scatter function calls.
        hist_kwargs : dict (optional)
            A dictionary of keyword arguments to pass to the
            matplotlib.pyplot.hist function calls.
    """

    M, N = data.shape

    if axes is None:
        skwargs = subplots_kwargs or {}
        skwargs.setdefault("sharex", False)
        skwargs.setdefault("sharey", False)
        skwargs.setdefault("figsize", (7, 6))
        fig, axes = plt.subplots(M, M, **skwargs)

    sc_kwargs = scatter_kwargs or {}
    sc_kwargs.setdefault("edgecolor", "none")
    sc_kwargs.setdefault("s", 10)

    hist_kwargs = hist_kwargs or {}
    hist_kwargs.setdefault("histtype", "stepfilled")
    hist_kwargs.setdefault("alpha", 0.8)

    xticks, yticks = None, None

    hist_color_idx = int(len(colors) / 2 - 1)
    if hist_color_idx < 0:
        hist_color_idx = 0
    hist_color = colors[hist_color_idx]

    icolors, colors_buff = iter(colors), {}

    for ii in range(M):
        for jj in range(M):
            ax = axes[ii, jj]
            col = (
                colors_buff[(ii, jj)]
                if (ii, jj) in colors_buff else
                next(icolors))
            if ii == jj:
                ax.hist(data[ii], color=hist_color, **hist_kwargs)
            else:
                ax.scatter(data[jj], data[ii], color=col, **sc_kwargs)
                colors_buff[(ii, jj)] = col
                colors_buff[(jj, ii)] = col

            if yticks is None:
                yticks = ax.get_yticks()[1: -1]

            if xticks is None:
                xticks = ax.get_xticks()[1: -1]

            # first column
            if jj == 0:
                ax.set_ylabel(labels[ii], rotation=10, labelpad=10)

                # Hack so ticklabels don't overlap
                ax.yaxis.set_ticks(yticks)

            # last row
            if ii == M - 1:
                ax.set_xlabel(labels[jj], rotation=10)

                # Hack so ticklabels don't overlap
                ax.xaxis.set_ticks(xticks)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    fig = axes[0, 0].figure
    fig.subplots_adjust(hspace=0.25, wspace=0.25, left=0.08,
                        bottom=0.08, top=0.9, right=0.9)
    return axes


def scmtx_plot(mtx, criteria, weights, anames, cnames,
               cmap=None, ax=None, **kwargs):

    colors = cmap(np.linspace(0, 1, mtx.shape[1] ** 2))

    return scatter_plot_matrix(
        mtx.T, labels=cnames, colors=colors, axes=ax, **kwargs)
