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

__doc__ = """test moora methods"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

from . import core

from .. import moora
from ..common import util


# =============================================================================
# BASE CLASS
# =============================================================================

class MooraTest(core.SKCriteriaTestCase):

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
            util.MAX, util.MIN, util.MAX
        ]
        self.rows = len(self.mtx)
        self.columns = len(self.mtx[0]) if self.rows else 0

    def test_ratio_with_weights(self):
        weights = [1, 1, 1, 1, 1, 1, 1]

        result = [5, 1, 3, 6, 4, 2]
        points = [-0.23206838, -0.03604841, -0.1209072,
                  -0.31909074, -0.16956892, -0.11065173]

        rank_result, points_result = moora.ratio(
            self.mtx, self.criteria, weights
        )

        self.assertAllClose(points_result, points, atol=1.e-4)
        self.assertAllClose(rank_result, result)

    def test_ratio(self):
        result = [5, 1, 3, 6, 4, 2]
        points = [-1.6245, -0.2523, -0.8464, -2.2336, -1.1870, -0.7746]

        rank_result, points_result = moora.ratio(self.mtx, self.criteria)

        self.assertAllClose(points_result, points, atol=1.e-4)
        self.assertAllClose(rank_result, result)

    def test_refpoint(self):
        result = [4, 5, 1, 6, 2, 3]
        points = [0.6893, 0.6999,  0.5982, 0.8597, 0.6002, 0.6148]

        rank_result, points_result = moora.refpoint(self.mtx, self.criteria)

        self.assertAllClose(points_result, points, atol=1.e-3)
        self.assertAllClose(rank_result, result)

    def test_refpoint_with_weights(self):
        weights = [1, 1, 1, 1, 1, 1, 1]
        result = [4, 5, 1, 6, 2, 3]
        points = [0.09847, 0.0999, 0.0854, 0.1227, 0.0857, 0.0878]

        rank_result, points_result = moora.refpoint(
            self.mtx, self.criteria, weights
        )

        self.assertAllClose(points_result, points, atol=1.e-3)
        self.assertAllClose(rank_result, result)

    def _test_fmf(self):
        result = [5, 1, 3, 6, 4, 2]
        points = [3.4343, 148689.356, 120.3441, 0.7882, 16.2917, 252.9155]

        rank_result, points_result = moora.fmf(self.mtx, self.criteria)

        self.assertAllClose(points_result, points, atol=1.e-4)
        self.assertAllClose(rank_result, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows-1),
                random.randint(0, self.columns-1)
            )
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0

        moora.fmf(self.mtx, self.criteria)

    def _test_fmf_only_max(self):
        self.criteria = [util.MAX] * len(self.criteria)

        result = [2, 6, 4, 1, 3, 5]
        points = [0.0011, 2.411e-08, 3.135e-05, 0.0037, 0.0002, 1.48e-05]

        rank_result, points_result = moora.fmf(self.mtx, self.criteria)

        self.assertAllClose(points_result, points, atol=1.e-4)
        self.assertAllClose(rank_result, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows-1),
                random.randint(0, self.columns-1)
            )
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0

        moora.fmf(self.mtx, self.criteria)

    def _test_fmf_only_min(self):
        self.criteria = [util.MIN] * len(self.criteria)

        result = [5, 1, 3, 6, 4, 2]
        points = [
            869.5146, 41476540.2, 31897.0622, 264.0502, 4171.5128, 67566.8851
        ]

        rank_result, points_result = moora.fmf(self.mtx, self.criteria)

        self.assertAllClose(points_result, points, atol=1.e-4)
        self.assertAllClose(rank_result, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows-1),
                random.randint(0, self.columns-1)
            )
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0

        moora.fmf(self.mtx, self.criteria)

    def _test_multimoora(self):
        result = [5, 1, 3, 6, 4, 2]
        mmora_mtx = [
            [5, 4, 5],
            [1, 5, 1],
            [3, 1, 3],
            [6, 6, 6],
            [4, 2, 4],
            [2, 3, 2]
        ]

        rank_result, mmora_mtx_result = moora.multimoora(
            self.mtx, self.criteria
        )

        self.assertAllClose(mmora_mtx_result, mmora_mtx, atol=1.e-4)
        self.assertAllClose(rank_result, result)
