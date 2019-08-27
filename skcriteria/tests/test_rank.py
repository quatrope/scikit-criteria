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

"""Test ranking procedures"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

from .. import rank

from .tcore import SKCriteriaTestCase


# =============================================================================
# TESTS
# =============================================================================

class RankTest(SKCriteriaTestCase):

    def setUp(self):
        super(RankTest, self).setUp()
        cols = random.randint(100, 1000)
        self.arr = [random.randint(-1000, 1000) for _ in range(cols)]

    def test_rank(self):
        manual = [None] * len(self.arr)
        for elem_idx, pos in enumerate(rank.rankdata(self.arr)):
            manual[int(pos) - 1] = self.arr[elem_idx]
        self.assertEqual(manual, sorted(self.arr))

    def test_rank_reverse(self):
        manual = [None] * len(self.arr)
        for elem_idx, pos in enumerate(rank.rankdata(self.arr, reverse=True)):
            manual[int(pos) - 1] = self.arr[elem_idx]
        self.assertEqual(manual, sorted(self.arr, reverse=True))


class Dominance(SKCriteriaTestCase):

    def test_dominance(self):
        # Data from:
        # Brauers, W. K. M., & Zavadskas, E. K. (2012).
        # Robustness of MULTIMOORA: a method for multi-objective optimization.
        # Informatica, 23(1), 1-25.

        a = [11, 20, 14]
        b = [14, 16, 15]
        c = [15, 19, 12]

        self.assertEqual(rank.dominance(a, b), 2)
        self.assertEqual(rank.dominance(b, a), -2)

        self.assertEqual(rank.dominance(b, c), 2)
        self.assertEqual(rank.dominance(c, b), -2)

        self.assertEqual(rank.dominance(c, a), 2)
        self.assertEqual(rank.dominance(a, c), -2)

        self.assertEqual(rank.dominance(a, a), 0)
        self.assertEqual(rank.dominance(b, b), 0)
        self.assertEqual(rank.dominance(c, c), 0)


class Equality(SKCriteriaTestCase):

    def test_equality(self):
        # Data from:
        # Brauers, W. K. M., & Zavadskas, E. K. (2012).
        # Robustness of MULTIMOORA: a method for multi-objective optimization.
        # Informatica, 23(1), 1-25.

        e, f = random.randint(0, 100), random.randint(0, 100)

        while e in (5, 7) or f in (5, 7) or e == f:
            e, f = random.randint(0, 100), random.randint(0, 100)

        a = [e, e, e]
        b = [5, e, 7]
        c = [5, f, 7]

        self.assertEqual(rank.equality(a, a), 3)
        self.assertEqual(rank.equality(b, b), 3)
        self.assertEqual(rank.equality(c, c), 3)

        self.assertEqual(rank.equality(a, b), 1)
        self.assertEqual(rank.equality(b, a), 1)

        self.assertEqual(rank.equality(b, c), 2)
        self.assertEqual(rank.equality(c, b), 2)

        self.assertEqual(rank.equality(a, c), 0)
        self.assertEqual(rank.equality(c, a), 0)


class KendalDominance(SKCriteriaTestCase):

    def test_kendall_dominance(self):
        # Data from:
        # Brauers, W. K. M., & Zavadskas, E. K. (2012).
        # Robustness of MULTIMOORA: a method for multi-objective optimization.
        # Informatica, 23(1), 1-25.

        a = [1, 1, 3]
        b = [3, 3, 1]
        c = [2, 2, 2]

        self.assertEqual(rank.kendall_dominance(a, b), (0, (5, 7)))
        self.assertEqual(rank.kendall_dominance(b, a), (1, (7, 5)))

        self.assertEqual(rank.kendall_dominance(a, c), (0, (5, 6)))
        self.assertEqual(rank.kendall_dominance(c, a), (1, (6, 5)))

        self.assertEqual(rank.kendall_dominance(c, b), (0, (6, 7)))
        self.assertEqual(rank.kendall_dominance(b, c), (1, (7, 6)))

        self.assertEqual(rank.kendall_dominance(a, a), (None, (5, 5)))
        self.assertEqual(rank.kendall_dominance(b, b), (None, (7, 7)))
        self.assertEqual(rank.kendall_dominance(c, c), (None, (6, 6)))


class SpearmianR(SKCriteriaTestCase):

    def test_spearmanr(self):
        # Data from:
        # Brauers, W. K. M., & Zavadskas, E. K. (2012).
        # Robustness of MULTIMOORA: a method for multi-objective optimization.
        # Informatica, 23(1), 1-25.

        expert_0 = [1, 2, 3, 4, 5, 6, 7]
        expert_1 = [7, 6, 5, 4, 3, 2, 1]
        caos = [random.randint(7, 100) for _ in range(len(expert_0))]

        self.assertEqual(rank.spearmanr(expert_0, expert_1), -1)
        self.assertEqual(rank.spearmanr(expert_1, expert_0), -1)

        self.assertEqual(rank.spearmanr(expert_0, expert_0), 1)
        self.assertEqual(rank.spearmanr(expert_1, expert_1), 1)

        self.assertNotIn(rank.spearmanr(caos, expert_0), (1, -1))
        self.assertNotIn(rank.spearmanr(expert_0, caos), (1, -1))

        self.assertNotIn(rank.spearmanr(caos, expert_1), (1, -1))
        self.assertNotIn(rank.spearmanr(expert_1, caos), (1, -1))

        self.assertEqual(rank.spearmanr(caos, caos), 1)
        self.assertTrue(rank.spearmanr(caos, list(reversed(caos))) < -1)
