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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """Test normalization functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================

import collections
import random

import numpy as np

import mock

from six.moves import range

from . import core

from .. import norm


# =============================================================================
# BASE
# =============================================================================

class NormTestBase(core.SKCriteriaTestCase):

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
                         mtx_result, arr_result, **kwargs):
        mtx_func_result = normfunc(self.mtx, axis=0)
        arr_func_result = normfunc(self.arr)
        self.assertAllClose(mtx_result, mtx_func_result, **kwargs)
        self.assertAllClose(arr_result, arr_func_result, **kwargs)

        mtx_func_result = norm.norm(normname, self.mtx, axis=0)
        arr_func_result = norm.norm(normname, self.arr)
        self.assertAllClose(mtx_result, mtx_func_result, **kwargs)
        self.assertAllClose(arr_result, arr_func_result, **kwargs)


# =============================================================================
# TESTS
# =============================================================================

class RegisterTest(core.SKCriteriaTestCase):

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
            [4,  5, 6]
        ]
        mtx_result = [
            [1, 0, 3],
            [4, 7, 6]
        ]

        self.arr = [1, -2, 3]
        arr_result = [3, 0, 5]
        self._test_normalizer(
            norm.push_negatives, "push_negatives", mtx_result, arr_result)


class Ass1to0(NormTestBase):
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


class NoneNorm(NormTestBase):

    def test_none(self):
        self._test_normalizer(norm.none, "none", self.mtx, self.arr)
