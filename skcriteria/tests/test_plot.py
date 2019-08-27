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

import numpy as np

import mock

from ..base import Data

from .tcore import SKCriteriaTestCase


# =============================================================================
# BASE
# =============================================================================

@mock.patch("matplotlib.pyplot.show")
class PlotTestCase(SKCriteriaTestCase):

    def setUp(self):
        self.alternative_n, self.criteria_n = 5, 3
        self.mtx = (
            np.random.rand(self.alternative_n, self.criteria_n) *
            np.random.choice([1, -1], (self.alternative_n, self.criteria_n)))
        self.criteria = np.random.choice([1, -1], self.criteria_n)
        self.weights = np.random.randint(1, 100, self.criteria_n)
        self.data = Data(self.mtx, self.criteria, self.weights)

    def test_invalid_name(self, *args):
        with self.assertRaises(ValueError):
            self.data.plot("fooo")

    def test_scatter(self, *args):
        self.data.plot("scatter")
        self.data.plot.scatter()

    def test_radar(self, *args):
        self.data.plot()
        self.data.plot("radar")
        self.data.plot.radar()

    def test_hist(self, *args):
        self.data.plot("hist")
        self.data.plot.hist()

    def test_box(self, *args):
        self.data.plot("box")
        self.data.plot.violin()

    def test_violin(self, *args):
        self.data.plot("violin")
        self.data.plot.violin()

    def test_bars(self, *args):
        self.data.plot("bars")
        self.data.plot.bars()
