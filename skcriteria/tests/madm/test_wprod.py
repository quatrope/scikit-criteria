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

"""test weighted product model"""


# =============================================================================
# IMPORTS
# =============================================================================

from .. import core
from ... import util, norm
from ...madm import wprod


# =============================================================================
# BASE CLASS
# =============================================================================

class WProdTest(core.SKCriteriaTestCase):
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
        self.criteria = [util.MAX, util.MAX, util.MAX, util.MAX]
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
        rank_result, points_result = wprod.wprod(*normdata)

        self.assertAllClose(rank_result, [1, 2, 3])
        self.assertAllClose(
            points_result, [-1.154253, -1.161619, -1.219155], atol=1.e-3)

    def test_wprod_min(self):
        self.criteria[0] = util.MIN

        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = wprod.wprod(*normdata)

        self.assertAllClose(rank_result, [2, 1, 3])
        self.assertAllClose(
            points_result, [-0.772, -0.4128, -0.9098], atol=1.e-3)

    def test_wprod_negative(self):
        self.mtx[0][0] = -self.mtx[0][0]
        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = wprod.wprod(*normdata)

        self.assertAllClose(rank_result, [3, 1, 2])
        self.assertAllClose(
            points_result, [-1.869671, -0.977075, -1.165967], atol=1.e-3)

    def test_wprod_zero(self):
        self.mtx[0][0] = 0

        normdata = self.normalize(self.mtx, self.criteria, self.weights)
        rank_result, points_result = wprod.wprod(*normdata)

        self.assertAllClose(rank_result, [3, 1, 2])
        self.assertAllClose(
            points_result, [-1.715391, -1.05992, -1.12996], atol=1.e-3)

    def test_wprod_dm(self):
        # Data From:
        # Weighted product model. (n.d.). Retrieved January 07, 2017,
        # from http://en.wikipedia.org/wiki/Weighted_product_model
        # this is the wikipedia example

        dm = wprod.WeightedProduct()
        decision = dm.decide(self.mtx, self.criteria, self.weights)

        self.assertAllClose(decision.rank_, [1, 2, 3])
        self.assertAllClose(
            decision.e_.points, [-1.154253, -1.161619, -1.219155], atol=1.e-3)
