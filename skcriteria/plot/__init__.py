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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """Plotting utilities

"""


# =============================================================================
# IMPORTS
# =============================================================================

import sys

import six

import numpy as np

from six.moves import zip

from matplotlib import cm

from .. import norm, util
from .radar import radar_plot
from .multihist import multihist_plot
from .scmtx import scmtx_plot
from .box import box_plot
from .violin import violin_plot


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

class PlotProxy(object):

    def __init__(self, data):
        self._data = data

    def __unicode__(self):
        return self.to_str()

    def __bytes__(self):
        encoding = sys.getdefaultencoding()
        return self.__unicode__().encode(encoding, 'replace')

    def __str__(self):
        """Return a string representation for a particular Object

        Invoked by str(df) in both py2/py3.
        Yields Bytestring in Py2, Unicode String in py3.
        """
        if six.PY3:
            return self.__unicode__()
        return self.__bytes__()

    def __repr__(self):
        return str(self)

    def __call__(self, plotname="radar", **kwargs):
        if plotname not in _plot_types:
            msg = "Invalid plotname '{}'. Chooce from: {}"
            raise ValueError(msg.format(plotname, ", ".join(_plot_types)))
        method = getattr(self, plotname)
        return method(**kwargs)

    def to_str(self):
        return "PlotProxy for {}".format(self._data)

    def preprocess(self, data, mnorm, wnorm, weighted,
                   show_criteria, **kwargs):
        # normalization
        nmtx = norm.norm(mnorm, data.mtx, criteria=data.criteria, axis=0)
        nweights = (
            norm.norm(wnorm, data.weights, criteria=data.criteria)
            if data.weights is not None else None)

        # weight the data
        if weighted and nweights is not None:
            wmtx = np.multiply(nmtx, nweights)
        else:
            wmtx = nmtx

        # labels for criteria
        criterias = (
            [" ({})".format(util.CRITERIA_STR[c]) for c in data.criteria]
            if show_criteria else
            [""] * len(data.criteria))

        if nweights is not None:
            clabels = [
                "{}{}\n(w.{:.2f})".format(cn, cr, cw)
                for cn, cr, cw in zip(data.cnames, criterias, nweights)]
        else:
            clabels = [
                "{}{}".format(cn, cr)
                for cn, cr in zip(data.cnames, criterias)]

        # color map parse
        kwargs["cmap"] = cm.get_cmap(name=kwargs.get("cmap"))

        kwargs.update({
            "mtx": wmtx, "criteria": data.criteria, "weights": nweights,
            "anames": kwargs.pop("anames", data.anames),
            "cnames": kwargs.pop("cnames", clabels)})
        return kwargs

    def plot(self, func, mnorm="none", wnorm="none",
             weighted=True, show_criteria=True, **kwargs):
        ppkwargs = self.preprocess(
            self._data, mnorm, wnorm, weighted, show_criteria, **kwargs)
        kwargs.update(ppkwargs)
        return func(**kwargs)

    @_plot_type
    def radar(self, **kwargs):
        return self.plot(radar_plot, show_criteria=False, **kwargs)

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
