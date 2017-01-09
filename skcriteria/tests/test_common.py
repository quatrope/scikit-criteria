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

__doc__ = """Test common functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================

import collections
import random

import numpy as np

from six.moves import range

from . import core

from .. import util, norm, rank


# =============================================================================
# cons.py TEST
# =============================================================================

class UtilTest(core.SKCriteriaTestCase):

    def setUp(self):
        super(UtilTest, self).setUp()
        self.min_max = (util.MIN, util.MAX)

    def test_criteriarr(self):
        # from list
        arr = [random.choice(self.min_max) for _ in self.rrange(100, 1000)]
        arr_result = util.criteriarr(arr)
        self.assertAllClose(arr, arr_result)
        self.assertIsInstance(arr_result, np.ndarray)

        # from array
        arr = np.array(
            [random.choice(self.min_max) for _ in self.rrange(100, 1000)]
        )
        arr_result = util.criteriarr(arr)
        self.assertAllClose(arr, arr_result)
        self.assertIsInstance(arr_result, np.ndarray)
        self.assertIs(arr, arr_result)

        # some fail
        arr = [
            random.choice(self.min_max) for _ in self.rrange(100, 1000)
        ] + [2]
        with self.assertRaises(ValueError):
            arr_result = util.criteriarr(arr)


# =============================================================================
# norm.py TEST
# =============================================================================

class NormTest(core.SKCriteriaTestCase):

    def setUp(self):
        super(NormTest, self).setUp()
        cols = random.randint(100, 1000)
        rows = random.randint(100, 1000)
        self.mtx = [
            [random.randint(1, 1000) for _ in range(cols)]
            for _ in range(rows)
        ]
        self.arr = [random.randint(1, 1000) for _ in range(cols)]

    def _test_normalizer(self, normfunc, mtx_result, arr_result, **kwargs):
        mtx_func_result = normfunc(self.mtx, axis=0)
        arr_func_result = normfunc(self.arr)
        self.assertAllClose(mtx_result, mtx_func_result, **kwargs)
        self.assertAllClose(arr_result, arr_func_result, **kwargs)

    def test_SumNormalizer(self):
        sums = collections.defaultdict(float)
        for row in self.mtx:
            for coln, col in enumerate(row):
                sums[coln] += col
        mtx_result = [
            [(col / sums[coln]) for coln, col in enumerate(row)]
            for row in self.mtx
        ]
        arr_sum = float(sum(self.arr))
        arr_result = [(col / arr_sum) for col in self.arr]
        self._test_normalizer(norm.sum, mtx_result, arr_result)

    def test_MaxNormalizer(self):
        maxes = collections.defaultdict(lambda: None)
        for row in self.mtx:
            for coln, col in enumerate(row):
                if maxes[coln] is None or maxes[coln] < col:
                    maxes[coln] = col
        mtx_result = [
            [(float(col) / maxes[coln]) for coln, col in enumerate(row)]
            for row in self.mtx
        ]
        arr_max = float(max(self.arr))
        arr_result = [(col / arr_max) for col in self.arr]
        self._test_normalizer(norm.max, mtx_result, arr_result)

    def test_VectorNormalizer(self):
        colsums = collections.defaultdict(float)
        for row in self.mtx:
            for coln, col in enumerate(row):
                colsums[coln] += col ** 2
        mtx_result = [
            [(col / np.sqrt(colsums[coln])) for coln, col in enumerate(row)]
            for row in self.mtx
        ]
        arr_sum = sum([col ** 2 for col in self.arr])
        arr_result = [(col / np.sqrt(arr_sum)) for col in self.arr]
        self._test_normalizer(norm.vector, mtx_result, arr_result)

    def test_push_negatives(self):
        self.mtx = [
            [1, -2, 3],
            [4,  5, 6]
        ]
        mtx_result = [
            [1, 0, 3],
            [4, 7, 6]
        ]

        self.arr = [1, -2, 3]
        arr_result = [3, 0, 5]
        self._test_normalizer(norm.push_negatives, mtx_result, arr_result)

    def test_add1to0(self):
        self.mtx = [
            [1, 0, 3],
            [4, 5, 0]
        ]

        mtx_result = [
            [1, 1, 4],
            [4, 6, 1],
        ]

        self.arr = [1, 0, 0]
        arr_result = [1, 1, 1]
        self._test_normalizer(norm.add1to0, mtx_result, arr_result, atol=1)


# =============================================================================
# RANK TEST
# =============================================================================

class RankTest(core.SKCriteriaTestCase):

    def setUp(self):
        super(RankTest, self).setUp()
        cols = random.randint(100, 1000)
        self.arr = [random.randint(-1000, 1000) for _ in range(cols)]

    def test_rank(self):
        manual = [None] * len(self.arr)
        for elem_idx, pos in enumerate(rank.rankdata(self.arr)):
            manual[int(pos) - 1] = self.arr[elem_idx]
        self.assertEquals(manual, sorted(self.arr))

    def test_rank_reverse(self):
        manual = [None] * len(self.arr)
        for elem_idx, pos in enumerate(rank.rankdata(self.arr, reverse=True)):
            manual[int(pos) - 1] = self.arr[elem_idx]
        self.assertEquals(manual, sorted(self.arr, reverse=True))

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
