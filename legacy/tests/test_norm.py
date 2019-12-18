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

"""Test normalization functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================

import collections
import random

import numpy as np

import mock

from .. import norm

from .tcore import SKCriteriaTestCase


# =============================================================================
# BASE
# =============================================================================

class NormTestBase(SKCriteriaTestCase):

    def setUp(self):
        super(NormTestBase, self).setUp()
        cols = random.randint(100, 1000)
        rows = random.randint(100, 1000)
        self.mtx = [
            [random.randint(1, 1000) for _ in range(cols)]
            for _ in range(rows)
        ]
        self.arr = [random.randint(1, 1000) for _ in range(cols)]

    def _test_normalizer(self, normfunc, normname,
                         mtx_result, arr_result, criteria=None, **kwargs):
        mtx_func_result = normfunc(self.mtx, axis=0, criteria=criteria)
        arr_func_result = normfunc(self.arr, criteria=criteria)
        self.assertAllClose(mtx_result, mtx_func_result, **kwargs)
        self.assertAllClose(arr_result, arr_func_result, **kwargs)

        mtx_func_result = norm.norm(
            normname, self.mtx, axis=0, criteria=criteria)
        arr_func_result = norm.norm(normname, self.arr, criteria=criteria)
        self.assertAllClose(mtx_result, mtx_func_result, **kwargs)
        self.assertAllClose(arr_result, arr_func_result, **kwargs)


# =============================================================================
# TESTS
# =============================================================================

class RegisterTest(SKCriteriaTestCase):

    @mock.patch("skcriteria.norm.NORMALIZERS", {})
    def test_register_as_function(self):
        def foo():
            pass

        norm.register("foo", foo)
        self.assertDictEqual(norm.NORMALIZERS, {"foo": foo})

    @mock.patch("skcriteria.norm.NORMALIZERS", {})
    def test_register_as_decorators(self):

        @norm.register("foo")
        def foo():
            pass

        self.assertDictEqual(norm.NORMALIZERS, {"foo": foo})

    def test_duplicated_normalizer(self):
        def foo():
            pass

        with self.assertRaises(norm.DuplicatedNameError):
            norm.register("vector", foo)

    def test_invalid_normalizer_name(self):
        with self.assertRaises(norm.NormalizerNotFound):
            norm.norm("foo", [1, 2, 3])


class SumNormalizer(NormTestBase):

    def test_sum(self):
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
        self._test_normalizer(norm.sum, "sum", mtx_result, arr_result)


class MaxNormalizer(NormTestBase):

    def test_max(self):
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
        self._test_normalizer(norm.max, "max", mtx_result, arr_result)


class VectorNormalizer(NormTestBase):

    def test_vector(self):
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
        self._test_normalizer(norm.vector, "vector", mtx_result, arr_result)


class PushNegatives(NormTestBase):

    def test_push_negatives(self):
        self.mtx = [
            [1, -2, 3],
            [4, 5, 6]
        ]
        mtx_result = [
            [1, 0, 3],
            [4, 7, 6]
        ]

        self.arr = [1, -2, 3]
        arr_result = [3, 0, 5]
        self._test_normalizer(
            norm.push_negatives, "push_negatives", mtx_result, arr_result)


class Add1to0(NormTestBase):

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
        self._test_normalizer(
            norm.add1to0, "add1to0", mtx_result, arr_result, atol=1)


class AddEpsto0(NormTestBase):

    def test_addEpsto0(self):
        self.mtx = [
            [1, 0, 3],
            [4, 5, 0]
        ]

        mtx_result = [
            [1, 0, 3],
            [4, 5, 0],
        ]

        self.arr = [1, 0, 0]
        arr_result = [1, 0, 0]
        self._test_normalizer(
            norm.addepsto0, "addepsto0", mtx_result, arr_result, atol=0.1)


class NoneNorm(NormTestBase):

    def test_none(self):
        self._test_normalizer(norm.none, "none", self.mtx, self.arr)


class IdealPoint(NormTestBase):

    def test_ideal_point(self):
        # Data from:
        # Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        # Determining objective weights in multiple criteria problems:
        # The critic method. Computers & Operations Research, 22(7), 763-770.

        self.mtx = [
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98]
        ]
        mtx_result = [
            [1., 0.21243523, 0.99702381],
            [0.57845188, 0., 1.],
            [0.53242678, 0.44559585, 0.46130952],
            [0.45606695, 0.78497409, 0.19940476],
            [0.41841004, 0.65025907, 0.40178571],
            [0.40376569, 1., 0.06845238],
            [0.29811715, 0.84455959, 0.33333333],
            [0., 0.79015544, 0.]
        ]
        self.arr = [61, 1.08, 4.33]
        arr_result = [1., 0., 0.05423899]
        self._test_normalizer(
            norm.ideal_point, "ideal_point",
            mtx_result, arr_result, criteria=[1, 1, 1])

    def test_ideal_point_axis_None(self):
        mtx = [1., 2., 3.]
        expected = [0, 0.5, 1]
        result = norm.ideal_point(mtx, criteria=[1, 1, 1], axis=None)
        self.assertAllClose(result, expected)

        expected = [1., 0.5, 0.]
        result = norm.ideal_point(mtx, criteria=[-1, -1, -1], axis=None)
        self.assertAllClose(result, expected)

        with self.assertRaises(ValueError):
            norm.ideal_point(mtx, criteria=[-1, -1, 1], axis=None)

    def test_ideal_point_axis_1(self):
        mtx = np.array([
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98]
        ]).T
        expected = np.array([
            [1., 0.21243523, 0.99702381],
            [0.57845188, 0., 1.],
            [0.53242678, 0.44559585, 0.46130952],
            [0.45606695, 0.78497409, 0.19940476],
            [0.41841004, 0.65025907, 0.40178571],
            [0.40376569, 1., 0.06845238],
            [0.29811715, 0.84455959, 0.33333333],
            [0., 0.79015544, 0.]
        ]).T
        result = norm.ideal_point(mtx, criteria=[1, 1, 1], axis=1)
        self.assertAllClose(result, expected)


class InvertMin(NormTestBase):

    def test_invert_min(self):
        # Data from:
        # Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
        # Determining objective weights in multiple criteria problems:
        # The critic method. Computers & Operations Research, 22(7), 763-770.

        self.mtx = [
            [61, 1.08, 4.33],
            [20.7, 0.26, 4.34],
            [16.3, 1.98, 2.53],
            [9, 3.29, 1.65],
            [5.4, 2.77, 2.33],
            [4, 4.12, 1.21],
            [-6.1, 3.52, 2.10],
            [-34.6, 3.31, 0.98]
        ]
        mtx_result = self.mtx

        self.arr = [61, 1.08, 4.33]
        arr_result = self.arr

        self._test_normalizer(
            norm.invert_min, "invert_min",
            mtx_result, arr_result, criteria=[1, 1, 1])

    def test_invert_min_axis_None(self):
        mtx = [
            [1., 2., 3.],
            [4., 5., 6.]
        ]
        expected = [
            [1, 0.5, 0.33],
            [0.25, 0.2, 0.16]
        ]

        result = norm.invert_min(mtx, criteria=[-1, -1, -1], axis=None)
        self.assertAllClose(result, expected, atol=0.1)
        result = norm.invert_min(mtx[0], criteria=[-1, -1, -1], axis=None)
        self.assertAllClose(result, expected[0], atol=0.1)

        with self.assertRaises(ValueError):
            norm.invert_min(mtx, criteria=[-1, -1, 1], axis=None)

    def test_invert_min_axis_1(self):
        mtx = [
            [1., 2., 3.],
            [4., 5., 6.]
        ]
        expected = [
            [1, 0.5, 0.33],
            [4., 5., 6.]
        ]

        result = norm.invert_min(mtx, criteria=[-1, 1], axis=1)
        self.assertAllClose(result, expected, atol=0.1)
