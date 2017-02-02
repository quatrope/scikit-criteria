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

__doc__ = """Test utilities"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

import numpy as np

from . import core

from .. import util


# =============================================================================
# TESTS
# =============================================================================

class Criteriarr(core.SKCriteriaTestCase):

    def setUp(self):
        super(Criteriarr, self).setUp()
        self.min_max = (util.MIN, util.MAX)

    def test_from_list(self):
        arr = [random.choice(self.min_max) for _ in self.rrange(100, 1000)]
        arr_result = util.criteriarr(arr)
        self.assertAllClose(arr, arr_result)
        self.assertIsInstance(arr_result, np.ndarray)

    def test_from_array(self):
        arr = np.array(
            [random.choice(self.min_max) for _ in self.rrange(100, 1000)])
        arr_result = util.criteriarr(arr)
        self.assertAllClose(arr, arr_result)
        self.assertIsInstance(arr_result, np.ndarray)
        self.assertIs(arr, arr_result)

    def test_no_min_max(self):
        arr = [
            random.choice(self.min_max) for _ in self.rrange(100, 1000)] + [2]
        with self.assertRaises(ValueError):
            util.criteriarr(arr)


class IsMtx(core.SKCriteriaTestCase):

    def setUp(self):
        self.cases = [
            ([[1, 2], [1, 2]], True),
            ([[1, 2], [1]], False),
            ([1], False),
            ([1, 2, 3], False),
            ([[1], [1]], True),
            ([], False)]

    def assertIsMtx(self, case, expected):
        result = util.is_mtx(case)
        if result != expected:
            msg = "{} must {} a matrix".format(
                case, "be" if expected else "not be")
            self.fail(msg)

    def test_as_list(self):
        for case, expected in self.cases:
            self.assertIsMtx(case, expected)

    def test_as_array(self):
        for case, expected in self.cases:
            case = np.asarray(case)
            self.assertIsMtx(case, expected)


class Nearest(core.SKCriteriaTestCase):

    def test_gt(self):
        arr = np.array([0.25, 0.27])
        result = util.nearest(arr, 0.26, "gt")
        self.assertAllClose(result, 0.27)

    def test_lt(self):
        arr = np.array([0.25, 0.27])
        result = util.nearest(arr, 0.26, "lt")
        self.assertAllClose(result, 0.25)
