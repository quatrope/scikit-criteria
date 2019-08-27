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

"""test topsis methods"""


# =============================================================================
# IMPORTS
# =============================================================================

from ...validate import MAX
from ...madm import closeness

from ..tcore import SKCriteriaTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class TopsisTest(SKCriteriaTestCase):
    mnorm = "vector"
    wnorm = "sum"

    def setUp(self):
        # Data From:
        # Tzeng, G. H., & Huang, J. J. (2011).
        # Multiple attribute decision making: methods and applications.
        # CRC press.
        self.mtx = [
            [5, 8, 4],
            [7, 6, 8],
            [8, 8, 6],
            [7, 4, 6],
        ]
        self.criteria = [MAX, MAX, MAX]

    def test_topsis(self):
        dm = closeness.TOPSIS()

        weights = [.3, .4, .3]

        result = [3, 2, 1, 4]
        points = [0.5037, 0.6581, 0.7482, 0.3340]

        decision = dm.decide(self.mtx, self.criteria, weights)

        self.assertAllClose(decision.e_.closeness, points, atol=1.e-4)
        self.assertAllClose(decision.rank_, result)
