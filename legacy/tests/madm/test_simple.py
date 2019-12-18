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

"""test simple methods"""


# =============================================================================
# IMPORTS
# =============================================================================

from ... import norm
from ...validate import MAX, MIN
from ...madm import simple

from ..tcore import SKCriteriaTestCase


# =============================================================================
# Tests
# =============================================================================

class WSumTest(SKCriteriaTestCase):
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
            MIN, MIN, MIN, MIN,
            MAX, MIN, MAX]
        self.weights = [20, 20, 20, 20, 20, 20, 20]

    def test_wsum(self):
        result = [5, 1, 3, 6, 4, 2]
        points = [3.531494, 1985.498321, 8.853048,
                  2.95696, 4.71996, 12.909456]

        dm = simple.WeightedSum()
        decision = dm.decide(self.mtx, self.criteria, self.weights)

        self.assertAllClose(decision.e_.points, points, atol=1.e-3)
        self.assertAllClose(decision.rank_, result)


class WProdTest(SKCriteriaTestCase):
    mnorm = "sum"
    wnorm = "sum"

    def setUp(self):
        # Data From:
        # Weighted product model. (n.d.). Retrieved January 07, 2017,
        # from http://en.wikipedia.org/wiki/Weighted_product_model

        self.mtx = [
            [25, 20, 15, 30],
            [10, 30, 20, 30],
            [30, 10, 30, 10],
        ]
        self.criteria = [MAX, MAX, MAX, MAX]
        self.weights = [20, 15, 40, 25]

    def normalize(self, mtx, criteria, weights):
        non_negative = norm.push_negatives(mtx, axis=0)
        non_zero = norm.add1to0(non_negative, axis=0)
        return super(WProdTest, self).normalize(
            non_zero, criteria, weights)

    def test_wprod(self):
        # Data From:
        # Weighted product model. (n.d.). Retrieved January 07, 2017,
        # from http://en.wikipedia.org/wiki/Weighted_product_model
        # this is the wikipedia example

        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = simple.wprod(*normdata)

        self.assertAllClose(rank_result, [1, 2, 3])
        self.assertAllClose(
            points_result, [-0.501286, -0.504485, -0.529472], atol=1.e-3)

    def test_wprod_min(self):
        self.criteria[0] = MIN

        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = simple.wprod(*normdata)

        self.assertAllClose(rank_result, [2, 1, 3])
        self.assertAllClose(
            points_result, [-0.335297, -0.179319, -0.395156], atol=1.e-3)

    def test_wprod_negative(self):
        self.mtx[0][0] = -self.mtx[0][0]
        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = simple.wprod(*normdata)

        self.assertAllClose(rank_result, [3, 1, 2])
        self.assertAllClose(
            points_result, [-0.811988, -0.424338, -0.506373], atol=1.e-3)

    def test_wprod_zero(self):
        self.mtx[0][0] = 0

        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = simple.wprod(*normdata)

        self.assertAllClose(rank_result, [3, 1, 2])
        self.assertAllClose(
            points_result, [-0.744985, -0.460317, -0.490735], atol=1.e-3)

    def test_wprod_dm(self):
        # Data From:
        # Weighted product model. (n.d.). Retrieved January 07, 2017,
        # from http://en.wikipedia.org/wiki/Weighted_product_model
        # this is the wikipedia example

        dm = simple.WeightedProduct()
        decision = dm.decide(self.mtx, self.criteria, self.weights)

        self.assertAllClose(decision.rank_, [1, 2, 3])
        self.assertAllClose(
            decision.e_.points, [-0.501286, -0.504485, -0.529472], atol=1.e-3)
