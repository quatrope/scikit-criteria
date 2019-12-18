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

__doc__ = """Test io functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

import numpy as np

from ..validate import MIN, MAX, criteriarr, is_mtx

from .tcore import SKCriteriaTestCase


# =============================================================================
# TEST
# =============================================================================

class Criteriarr(SKCriteriaTestCase):

    def setUp(self):
        super(Criteriarr, self).setUp()
        self.min_max = (MIN, MAX)

    def test_from_list(self):
        arr = [random.choice(self.min_max) for _ in self.rrange(100, 1000)]
        arr_result = criteriarr(arr)
        self.assertAllClose(arr, arr_result)
        self.assertIsInstance(arr_result, np.ndarray)

    def test_no_min_max(self):
        arr = [
            random.choice(self.min_max) for _ in self.rrange(100, 1000)] + [2]
        with self.assertRaises(ValueError):
            criteriarr(arr)

    def test_alias(self):
        original = np.array([-1, 1])
        criterias = (
            [min, max],
            [np.min, np.max],
            [np.amin, np.amax],
            [np.nanmin, np.nanmax],
            ["minimize", "maximize"],
            ["min", "max"])
        for arr in criterias:
            parsed = criteriarr(arr)
            self.assertArrayEqual(parsed, original)
            with self.assertRaises(ValueError):
                criteriarr(arr + [2])
            with self.assertRaises(ValueError):
                criteriarr(arr + ["foo"])


class IsMtx(SKCriteriaTestCase):

    def setUp(self):
        self.cases = [
            ([[1, 2], [1, 2]], True),
            ([[1, 2], [1]], False),
            ([1], False),
            ([1, 2, 3], False),
            ([[1], [1]], True),
            ([], False)]

    def assertIsMtx(self, case, expected):
        result = is_mtx(case)
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
