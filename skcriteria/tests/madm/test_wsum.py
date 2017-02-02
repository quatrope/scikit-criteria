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
# DOC
# =============================================================================

__doc__ = """test moora methods"""


# =============================================================================
# IMPORTS
# =============================================================================

from .. import core
from ... import util
from ...madm import wsum


# =============================================================================
# BASE CLASS
# =============================================================================

class WSumTest(core.SKCriteriaTestCase):
    mnorm = "sum"
    wnorm = "sum"

    def setUp(self):
        # Data From:
        # KRACKA, M; BRAUERS, W. K. M.; ZAVADSKAS, E. K. Ranking
        # Heating Losses in a Building by Applying the MULTIMOORA . -
        # ISSN 1392 – 2785 Inzinerine Ekonomika-Engineering Economics, 2010,
        # 21(4), 352-359.

        self.mtx = [
            [33.95, 23.78, 11.45, 39.97, 29.44, 167.10, 3.852],
            [38.9, 4.17, 6.32, 0.01, 4.29, 132.52, 25.184],
            [37.59, 9.36, 8.23, 4.35, 10.22, 136.71, 10.845],
            [30.44, 37.59, 13.91, 74.08, 45.10, 198.34, 2.186],
            [36.21, 14.79, 9.17, 17.77, 17.06, 148.3, 6.610],
            [37.8, 8.55, 7.97, 2.35, 9.25, 134.83, 11.935]
        ]
        self.criteria = [
            util.MIN, util.MIN, util.MIN, util.MIN,
            util.MAX, util.MIN, util.MAX]

    def test_mdwsum_with_weights(self):
        weights = [20, 20, 20, 20, 20, 20, 20]

        result = [5,  1,  3,  6,  4,  2]
        points = [-0.1075, -0.0037, -0.0468, -0.1560, -0.0732, -0.0413]

        normdata = self.normalize(self.mtx, self.criteria, weights)
        rank_result, points_result = wsum.mdwsum(*normdata)

        self.assertAllClose(points_result, points, atol=1.e-3)
        self.assertAllClose(rank_result, result)

    def test_mdwsum(self):
        result = [5,  1,  3,  6,  4,  2]
        points = [-0.7526, -0.026, -0.3273, -1.092, -0.5127, -0.2894]

        normdata = self.normalize(self.mtx, self.criteria, weights=None)
        rank_result, points_result = wsum.mdwsum(*normdata)

        self.assertAllClose(points_result, points, atol=1.e-3)
        self.assertAllClose(rank_result, result)

    def test_mdwsum_dm(self):
        weights = [20, 20, 20, 20, 20, 20, 20]

        result = [5,  1,  3,  6,  4,  2]
        points = [-0.1075, -0.0037, -0.0468, -0.1560, -0.0732, -0.0413]

        dm = wsum.MDWeightedSum()
        decision = dm.decide(self.mtx, self.criteria, weights)

        self.assertAllClose(decision.e_.points, points, atol=1.e-3)
        self.assertAllClose(decision.rank_, result)
