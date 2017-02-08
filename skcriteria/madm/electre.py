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


__doc__ = """

Data from Tzeng and Huang et al, 2011 [TZENG2011]_

References
----------

.. [TZENG2011] Tzeng, G. H., & Huang, J. J. (2011). Multiple
   attribute decision making: methods and applications. CRC press.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import util
from ..dmaker import DecisionMaker


# =============================================================================
# UTILS
# =============================================================================

def concordance(mtx, criteria, weights):

    mtx_criteria = np.tile(criteria, (len(mtx), 1))
    mtx_weight = np.tile(weights, (len(mtx), 1))
    mtx_concordance = np.empty((len(mtx), len(mtx)))

    for idx, row in enumerate(mtx):
        difference = row - mtx
        outrank = (
            ((mtx_criteria == util.MAX) & (difference >= 0)) |
            ((mtx_criteria == util.MIN) & (difference <= 0))
        )
        filter_weights = mtx_weight * outrank.astype(int)
        new_row = np.sum(filter_weights, axis=1)
        mtx_concordance[idx] = new_row

    np.fill_diagonal(mtx_concordance, np.nan)
    return mtx_concordance


def discordance(mtx, criteria):

    mtx_criteria = np.tile(criteria, (len(mtx), 1))
    mtx_discordance = np.empty((len(mtx), len(mtx)))
    ranges = np.max(mtx, axis=0) - np.min(mtx, axis=0)

    for idx, row in enumerate(mtx):
        difference = mtx - row
        worsts = (
            ((mtx_criteria == util.MAX) & (difference > 0)) |
            ((mtx_criteria == util.MIN) & (difference < 0))
        )
        filter_difference = np.abs(difference * worsts)
        delta = filter_difference / ranges
        new_row = np.max(delta, axis=1)
        mtx_discordance[idx] = new_row

    np.fill_diagonal(mtx_discordance, np.nan)
    return mtx_discordance


# =============================================================================
# ELECTRE
# =============================================================================

def electre1(nmtx, ncriteria, nweights, p, q):

    # get the concordance and discordance info
    mtx_concordance = concordance(nmtx, ncriteria, nweights)
    mtx_discordance = discordance(nmtx, ncriteria)

    with np.errstate(invalid='ignore'):
        outrank = (
            (mtx_concordance > p) & (mtx_discordance < q))

    bt_rows = np.sum(outrank, axis=1)
    bt_columns = np.sum(outrank, axis=0)

    diff = bt_rows - bt_columns
    max_value = np.max(diff)

    kernel = np.where((diff == max_value) & (diff > 0))[0]

    return kernel, outrank, mtx_concordance, mtx_discordance


# =============================================================================
# OO
# =============================================================================

class ELECTRE1(DecisionMaker):

    def __init__(self, p=.65, q=.35, mnorm="sum", wnorm="sum"):
        super(ELECTRE1, self).__init__(mnorm=mnorm, wnorm=wnorm)
        self._p = float(p)
        self._q = float(q)

    def solve(self, nmtx, ncriteria, nweights):
        kernel, outrank, mtx_concordance, mtx_discordance = electre1(
            nmtx=nmtx, ncriteria=ncriteria, nweights=nweights,
            p=self._p, q=self._q)

        extra = {
            "outrank": outrank,
            "mtx_concordance": mtx_concordance,
            "mtx_discordance": mtx_discordance,
            "p": self.p, "q": self.q}

        return kernel, None, extra

    def as_dict(self):
        base = super(ELECTRE1, self).as_dict()
        base.update({"p": self._p, "q": self._q})
        return base

    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q
