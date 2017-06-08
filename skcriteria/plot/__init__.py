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

from .. import norm

from .radar import radar_plot


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
        method = getattr(self, plotname)
        return method(**kwargs)

    def to_str(self):
        return "PlotProxy for {}".format(self._data)

    def preprocess(self, data, mnorm, wnorm):
        nmtx = norm.norm(mnorm, data.mtx, criteria=data.criteria, axis=0)
        nweights = (
            norm.norm(wnorm, data.weights, criteria=data.criteria)
            if data.weights is not None else
            np.ones(data.criteria.shape))
        return nmtx, data.criteria, nweights

    def plot(self, func, mnorm="none", wnorm="none", **kwargs):
        data = self._data
        nmtx, criteria, nweights = self.preprocess(data, mnorm, wnorm)
        kwargs.update({
            "mtx": nmtx, "criteria": criteria,
            "weights": nweights, "anames": data.anames,
            "cnames": data.cnames})
        return func(**kwargs)

    def radar(self, **kwargs):
        self.plot(radar_plot, **kwargs)
