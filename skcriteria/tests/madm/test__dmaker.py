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

"""test electre methods"""


# =============================================================================
# IMPORTS
# =============================================================================

import random
import string

from ...madm import _dmaker

from ..tcore import SKCriteriaTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class ExtraTest(SKCriteriaTestCase):

    def setUp(self):
        self.data = {}
        for idx in range(random.randint(10, 100)):
            key = "".join([
                random.choice(string.ascii_letters)
                for _ in range(random.randint(10, 30))])
            value = "".join([
                random.choice(string.ascii_letters)
                for _ in range(random.randint(10, 30))])
            self.data[key + str(idx)] = value
        self.e = _dmaker.Extra(self.data)

    def test_eq(self):
        self.assertTrue(self.e == _dmaker.Extra(self.data))

    def test_ne(self):
        e = self.e
        self.setUp()
        self.assertTrue(self.e != e)

    def test_getitem(self):
        for k, v in self.data.items():
            self.assertEqual(self.e[k], v)

    def test_iter(self):
        for k in self.e:
            self.assertIn(k, self.data)

    def test_len(self):
        self.assertEqual(len(self.data), len(self.e))

    def test_getattr(self):
        for k, v in self.data.items():
            self.assertEqual(getattr(self.e, k), v)

    def test_str(self):
        str(self.e)

    def test_repr(self):
        repr(self.e)
