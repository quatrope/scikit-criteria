#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


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
from ... import util
from ...madm import wprod


# =============================================================================
# BASE CLASS
# =============================================================================

class WProdTest(core.SKCriteriaTestCase):

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

    def test_wprod(self):
        # Data From:
        # Weighted product model. (n.d.). Retrieved January 07, 2017,
        # from http://en.wikipedia.org/wiki/Weighted_product_model
        # this is the wikipedia example

        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights)

        self.assertAllClose(rank_result, [1, 2, 3])
        self.assertAllClose(
            points_result, [-1.154253, -1.161619, -1.219155], atol=1.e-3)

    def test_wprod_min(self):
        self.criteria[0] = util.MIN

        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights)

        self.assertAllClose(rank_result, [2, 1, 3])
        self.assertAllClose(
            points_result, [-0.772, -0.4128, -0.9098], atol=1.e-3)

    def test_wprod_negative(self):
        self.mtx[0][0] = -self.mtx[0][0]
        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights)

        self.assertAllClose(rank_result, [3, 1, 2])
        self.assertAllClose(
            points_result, [-1.869671, -0.977075, -1.165967], atol=1.e-3)

    def test_wprod_zero(self):
        self.mtx[0][0] = 0

        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights)

        self.assertAllClose(rank_result, [3, 1, 2])
        self.assertAllClose(
            points_result, [-1.715391, -1.05992, -1.12996], atol=1.e-3)
