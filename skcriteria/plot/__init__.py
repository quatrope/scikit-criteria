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

from ..core import MIN, CRITERIA_STR
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

    def preprocess(self, data, mnorm, wnorm,
                   weighted, show_criteria,
                   min2max, push_negatives,
                   addepsto0, **kwargs):

        # extract all the data
        mtx, criteria, weights, cnames, anames = (
            data.mtx, data.criteria, data.weights, data.cnames, data.anames)

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

        # color map parse
        kwargs["cmap"] = cm.get_cmap(name=kwargs.get("cmap"))

        kwargs.update({
            "mtx": mtx, "criteria": criteria, "weights": weights,
            "anames": kwargs.pop("anames", anames),
            "cnames": kwargs.pop("cnames", cnames)})
        return kwargs

    def plot(self, func, mnorm="none", wnorm="none",
             weighted=True, show_criteria=True, addepsto0=False,
             min2max=False, push_negatives=False, **kwargs):
        ppkwargs = self.preprocess(
            data=self._data, mnorm=mnorm, wnorm=wnorm,
            weighted=weighted, show_criteria=show_criteria,
            addepsto0=addepsto0, min2max=min2max,
            push_negatives=push_negatives, **kwargs)
        kwargs.update(ppkwargs)
        return func(**kwargs)

    @_plot_type
    def radar(self, **kwargs):
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
