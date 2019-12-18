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
# DOC
# =============================================================================

"""Test normalization functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================


from ... import norm, divcorr
from ...base import Data
from ...weights import divergence

from ..tcore import SKCriteriaTestCase


# =============================================================================
# BASE
# =============================================================================

class DivergenceTest(SKCriteriaTestCase):

    def setUp(self):
        # Data from:
        # Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        # Determining objective weights in multiple criteria problems:
        # The critic method. Computers & Operations Research, 22(7), 763-770.
        self.mtx = [
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98]
        ]
        self.nmtx = norm.ideal_point(self.mtx, criteria=[1, 1, 1], axis=0)
        self.expected = [0.27908306, 0.34092628, 0.37999065]

    def test_divergence(self):
        result = divergence.divergence(self.nmtx, divcorr.std)
        self.assertAllClose(result, self.expected)

    def test_divergence_oop(self):
        data = Data(self.mtx, [1, 1, 1])
        wd = divergence.DivergenceWeights()
        rdata = wd.decide(data)
        self.assertAllClose(rdata.weights, self.expected)
