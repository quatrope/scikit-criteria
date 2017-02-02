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
# IMPORTS
# =============================================================================

import numpy as np

from ..import util, rank
from ..dmaker import DecisionMaker


# =============================================================================
# Function
# =============================================================================

def topsis(nmtx, ncriteria, nweights):

    # apply weights
    wmtx = np.multiply(nmtx, nweights)

    # extract mins and maxes
    mins = np.min(wmtx, axis=0)
    maxs = np.max(wmtx, axis=0)

    # create the ideal and the anti ideal arrays
    ideal = np.where(ncriteria == util.MAX, maxs, mins)
    anti_ideal = np.where(ncriteria == util.MIN, maxs, mins)

    # calculate distances
    d_better = np.sqrt(np.sum(np.power(wmtx - ideal, 2), axis=1))
    d_worst = np.sqrt(np.sum(np.power(wmtx - anti_ideal, 2), axis=1))

    # relative closeness
    closeness = d_worst / (d_better + d_worst)

    # compute the rank and return the result
    return rank.rankdata(closeness, reverse=True), closeness


# =============================================================================
# OO
# =============================================================================

class TOPSIS(DecisionMaker):

    def __init__(self, mnorm="vector", wnorm="sum"):
        super(TOPSIS, self).__init__(mnorm=mnorm, wnorm=wnorm)

    def solve(self, *args, **kwargs):
        rank, closeness = topsis(*args, **kwargs)
        return None, rank, {"closeness": closeness}
