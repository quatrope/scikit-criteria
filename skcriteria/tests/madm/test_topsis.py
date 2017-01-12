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

__doc__ = """test topsis methods"""


# =============================================================================
# IMPORTS
# =============================================================================

from .. import core
from ... import util
from ...madm import topsis


# =============================================================================
# BASE CLASS
# =============================================================================

class TopsisTest(core.SKCriteriaTestCase):

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
        self.criteria = [util.MAX, util.MAX, util.MAX]

    def test_topsis(self):
        weights = [.3, .4, .3]

        result = [3, 2, 1, 4]
        points = [0.5037, 0.6581, 0.7482, 0.3340]

        rank_result, points_result = topsis.topsis(
            self.mtx, self.criteria, weights)

        self.assertAllClose(points_result, points, atol=1.e-4)
        self.assertAllClose(rank_result, result)

    def test_topsis_dm(self):
        dm = topsis.TOPSIS()

        weights = [.3, .4, .3]

        result = [3, 2, 1, 4]
        points = [0.5037, 0.6581, 0.7482, 0.3340]

        decision = dm.decide(self.mtx, self.criteria, weights)

        self.assertAllClose(decision.points_, points, atol=1.e-4)
        self.assertAllClose(decision.rank_, result)
