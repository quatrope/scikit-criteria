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

"""Estimate the weights of every criteria bye some kind of divergence index

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ._wdeterminer import WeightDeterminer
from ..divcorr import DIVERGENCE_FUNCTIONS


# =============================================================================
# WEIGHTS
# =============================================================================

def divergence(nmtx, dfunction):
    dindex = dfunction(nmtx)
    return dindex / np.sum(dindex)


class DivergenceWeights(WeightDeterminer):

    def __init__(self, dfunction="std", mnorm="ideal_point"):
        super(DivergenceWeights, self).__init__(mnorm=mnorm)
        self._dfunction = DIVERGENCE_FUNCTIONS.get(dfunction, dfunction)
        if not hasattr(self._dfunction, "__call__"):
            msg = "'dfunction' must be a callable or a string in {}. Found {}"
            raise TypeError(msg.format(DIVERGENCE_FUNCTIONS.keys(), dfunction))

    def as_dict(self):
        data = super(DivergenceWeights, self).as_dict()
        data.update({"dfunction": self._dfunction.__name__})
        return data

    def solve(self, ndata):
        nmtx = ndata.mtx
        weights = divergence(nmtx, self._dfunction)
        return weights,
