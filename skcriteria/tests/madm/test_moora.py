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

"""test moora methods"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

import numpy as np

from ... import norm
from ...validate import MIN, MAX
from ...madm import moora

from ..tcore import SKCriteriaTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class MOORATestBase(SKCriteriaTestCase):
    mnorm = "vector"
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
            MAX, MIN, MAX
        ]
        self.rows = len(self.mtx)
        self.columns = len(self.mtx[0]) if self.rows else 0


# =============================================================================
# TESTS
# =============================================================================

class RatioMOORATest(MOORATestBase):

    def test_ratio(self):
        weights = [1, 1, 1, 1, 1, 1, 1]

        result = [5, 1, 3, 6, 4, 2]
        points = [
            -0.232068, -0.036048, -0.120907, -0.319091, -0.169569, -0.110652]

        dm = moora.RatioMOORA()
        decision = dm.decide(self.mtx, self.criteria, weights)

        self.assertAllClose(decision.e_.points, points, atol=1.e-4)
        self.assertAllClose(decision.rank_, result)


class RefPointMOORATest(MOORATestBase):

    def test_refpoint(self):
        weights = [1, 1, 1, 1, 1, 1, 1]
        result = [4, 5, 1, 6, 2, 3]
        points = [0.09847, 0.0999, 0.0854, 0.1227, 0.0857, 0.0878]

        dm = moora.RefPointMOORA()
        decision = dm.decide(self.mtx, self.criteria, weights)

        self.assertAllClose(decision.e_.points, points, atol=1.e-4)
        self.assertAllClose(decision.rank_, result)


class FMFMOORATest(MOORATestBase):

    def normalize(self, mtx, criteria):
        non_negative = norm.push_negatives(mtx, axis=0)
        non_zero = norm.add1to0(non_negative, axis=0)
        nmtx, ncriteria, nweights = super(FMFMOORATest, self).normalize(
            non_zero, criteria, 1)
        return nmtx, ncriteria

    def test_fmf(self):
        result = [5, 1, 3, 6, 4, 2]

        # the result is the logarithm of this values
        points = [3.4343, 148689.356, 120.3441, 0.7882, 16.2917, 252.9155]

        dm = moora.FMFMOORA()
        decision = dm.decide(self.mtx, self.criteria)

        self.assertAllClose(decision.e_.points, np.log(points), atol=1.e-4)
        self.assertAllClose(decision.rank_, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows - 1),
                random.randint(0, self.columns - 1))
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0
        dm.decide(self.mtx, self.criteria)

    def test_fmf_only_max(self):
        self.criteria = [MAX] * len(self.criteria)

        result = [2, 6, 4, 1, 3, 5]

        # the result is the logarithm of this values
        points = [0.0011, 2.411e-08, 3.135e-05, 0.0037, 0.0002, 1.48e-05]

        dm = moora.FMFMOORA()
        decision = dm.decide(self.mtx, self.criteria)

        self.assertAllClose(decision.e_.points, np.log(points), atol=1.)
        self.assertAllClose(decision.rank_, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows - 1),
                random.randint(0, self.columns - 1))
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0
        dm.decide(self.mtx, self.criteria)

    def test_fmf_only_min(self):

        self.criteria = [MIN] * len(self.criteria)

        result = [5, 1, 3, 6, 4, 2]

        # the result is the logarithm + 1 of this values
        points = [
            869.5146, 41476540.2, 31897.0622, 264.0502, 4171.5128, 67566.8851]

        dm = moora.FMFMOORA()
        decision = dm.decide(self.mtx, self.criteria)

        self.assertAllClose(decision.e_.points, 1 + np.log(points), atol=1.e-4)
        self.assertAllClose(decision.rank_, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows - 1),
                random.randint(0, self.columns - 1))
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0
        dm.decide(self.mtx, self.criteria)


class MultiMOORATest(MOORATestBase):

    def normalize(self, mtx, criteria):
        non_negative = norm.push_negatives(mtx, axis=0)
        non_zero = norm.add1to0(non_negative, axis=0)
        nmtx, ncriteria, nweights = super(MultiMOORATest, self).normalize(
            non_zero, criteria, 1)
        return nmtx, ncriteria

    def test_multimoora(self):
        result = [5, 1, 3, 6, 4, 2]
        mmora_mtx = [
            [5, 4, 5],
            [1, 5, 1],
            [3, 1, 3],
            [6, 6, 6],
            [4, 2, 4],
            [2, 3, 2]
        ]

        dm = moora.MultiMOORA()
        decision = dm.decide(self.mtx, self.criteria)

        self.assertAllClose(decision.e_.rank_mtx, mmora_mtx, atol=1.e-4)
        self.assertAllClose(decision.rank_, result)

        # some zeroes
        zeros = set()
        while len(zeros) < 3:
            zero = (
                random.randint(0, self.rows - 1),
                random.randint(0, self.columns - 1))
            zeros.add(zero)
        for row, column in zeros:
            self.mtx[row][column] = 0
        dm.decide(self.mtx, self.criteria)
