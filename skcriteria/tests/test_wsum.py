#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> and <nluczywo@gmail.com>
# wrote this file. As long as you retain this notice you can do whatever you
# want with this stuff. If we meet some day, and you think this stuff is worth
# it, you can buy me a WISKEY in return Juan BC and Nadia AL.


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals

# =============================================================================
# DOC
# =============================================================================

"""test moora methods"""


# =============================================================================
# IMPORTS
# =============================================================================

from . import core

from .. import wsum
from ..common import util


# =============================================================================
# BASE CLASS
# =============================================================================

class WSumTest(core.SKCriteriaTestCase):

    def setUp(self):
        """Data from Kracka et al, 2010 [KRACKA2010]_

        References
        ----------

        .. [KRACKA2010] KRACKA, M; BRAUERS, W. K. M.; ZAVADSKAS, E. K. Ranking
           Heating Losses in a Building by Applying the MULTIMOORA . -
           ISSN 1392 – 2785 Inzinerine Ekonomika-Engineering Economics, 2010,
           21(4), 352-359.

        """
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

    def test_mdwsum_with_weights(self):
        weights = [20, 20, 20, 20, 20, 20, 20]

        result = [5,  1,  3,  6,  4,  2]
        points = [-0.1075, -0.0037, -0.0468, -0.1560, -0.0732, -0.0413]

        rank_result, points_result = wsum.mdwsum(
            self.mtx, self.criteria, weights
        )

        self.assertIsClose(points_result, points, atol=1.e-3)
        self.assertIsClose(rank_result, result)

    def test_mdwsum(self):
        result = [5,  1,  3,  6,  4,  2]
        points = [-0.7526, -0.026, -0.3273, -1.092, -0.5127, -0.2894]

        rank_result, points_result = wsum.mdwsum(self.mtx, self.criteria)

        self.assertIsClose(points_result, points, atol=1.e-3)
        self.assertIsClose(rank_result, result)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
