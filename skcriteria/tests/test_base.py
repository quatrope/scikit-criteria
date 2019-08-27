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

"""Test base.Data functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

import numpy as np

from ..base import Data
from .tcore import SKCriteriaTestCase


# =============================================================================
# TEST
# =============================================================================

class TestData(SKCriteriaTestCase):

    def setUp(self):
        self.mtx = np.random.rand(3, 3)
        self.criteria = np.asarray([random.choice((1, -1)) for e in range(3)])
        self.weights = np.random.randint(1, 100, 3)

    def test_trivial(self):
        data = Data(self.mtx, self.criteria, self.weights)
        other = Data(self.mtx, self.criteria, self.weights)
        self.assertEqual(data, other)

    def test_metadata(self):
        meta = {"foo": "faa"}
        data = Data(self.mtx, self.criteria, self.weights, meta=meta)
        self.assertEqual(data.meta, meta)
        self.assertEqual(data.meta.foo, data.meta["foo"])
        with self.assertRaises(KeyError):
            data.meta["fee"]
        with self.assertRaises(AttributeError):
            data.meta.fee
        data.meta.to_str()

    def test_unicode_string(self):
        cnames = anames = u"ñññ"
        data = Data(
            self.mtx, self.criteria, self.weights,
            anames=anames, cnames=cnames)
        data.to_str()
        data.to_str(tablefmt="html")

        repr(data)
        str(data)
