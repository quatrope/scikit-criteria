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

from . import core

from .. import wprod
from ..common import util


# =============================================================================
# BASE CLASS
# =============================================================================

class WProdTest(core.SKCriteriaTestCase):

    def setUp(self):
        """Data from
         `Wikipedia <http://en.wikipedia.org/wiki/Weighted_product_model>`_

        """
        self.mtx = [
            [25, 20, 15, 30],
            [10, 30, 20, 30],
            [30, 10, 30, 10],
        ]
        self.criteria = [util.MAX, util.MAX, util.MAX, util.MAX]
        self.weights = [20, 15, 40, 25]

    def test_wprod(self):
        # this is the wikipedia example
        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights
        )

        self.assertIsClose(
            points_result[0]/points_result[1], 1.007, atol=1.e-3
        )
        self.assertIsClose(
            points_result[0]/points_result[2], 1.067, atol=1.e-3
        )
        self.assertIsClose(
            points_result[1]/points_result[2], 1.059, atol=1.e-3
        )
        self.assertIsClose(rank_result, [1, 2, 3])

    def test_wprod_min(self):
        self.criteria[0] = util.MIN

        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights
        )

        self.assertIsClose(rank_result, [2, 1, 3])
        self.assertIsClose(points_result, [0.2847, 0.4079, 0.248], atol=1.e-3)

    def test_wprod_negative(self):
        self.mtx[0][0] = -self.mtx[0][0]

        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights
        )

        self.assertIsClose(rank_result, [3, 1, 2])
        self.assertIsClose(points_result, [0, 0.3768, 0.3125], atol=1.e-3)

    def test_wprod_zero(self):
        self.mtx[0][0] = 0

        rank_result, points_result = wprod.wprod(
            self.mtx, self.criteria, self.weights
        )

        self.assertIsClose(rank_result, [3, 1, 2])
        self.assertIsClose(points_result, [0.0001, 0.3449, 0.3256], atol=1.e-3)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
