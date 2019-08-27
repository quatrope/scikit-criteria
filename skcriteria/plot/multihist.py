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


"""
Extracted from: http://matplotlib.org/examples/style_sheets/plot_bmh.html

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import matplotlib.pyplot as plt


# =============================================================================
# FUNCTIONS
# =============================================================================

def multihist_plot(
        mtx, criteria, weights, anames, cnames,
        cmap=None, ax=None, subplots_kwargs=None, hist_kwargs=None):

    # create ax if necesary
    if ax is None:
        subplots_kwargs = subplots_kwargs or {}
        subplots_kwargs.setdefault("figsize", (7, 6))
        ax = plt.subplots(**subplots_kwargs)[-1]

    # colors
    colors = cmap(np.linspace(0, 1, mtx.shape[1]))

    # histogram
    hist_kwargs = hist_kwargs or {}
    hist_kwargs.setdefault("histtype", "stepfilled")
    hist_kwargs.setdefault("alpha", 0.8)

    for arr, col in zip(mtx.T, colors):
        ax.hist(arr, color=col, **hist_kwargs)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend(cnames, loc="best")

    return ax
