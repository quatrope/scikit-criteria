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
# DOCS
# =============================================================================

"""test integration with recipe
http://code.activestate.com/recipes/577748/ test case

"""


# =============================================================================
# IMPORTS
# =============================================================================

from ...utils import lp

from ..tcore import SKCriteriaTestCase


# =============================================================================
# TESTS
# =============================================================================

class LPTest(SKCriteriaTestCase):

    def test_maximize_frommtx(self):
        model = lp.Maximize.frommtx(
            c=[250, 130, 350],
            A=[
                [120, 200, 340],
                [-20, -40, -15],
                [800, 1000, 600]
            ],
            b=[500, -15, 1000])
        result = model.solve()
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.objective, 540)
        self.assertEqual(result.variables, ('x1', 'x2', 'x3'))
        self.assertEqual(result.values, (0.2, 0.0, 1.4))

    def test_minimize_frommtx(self):
        model = lp.Minimize.frommtx(
            c=[250, 130, 350],
            A=[
                [120, 200, 340],
                [20, 40, 15],
                [800, 1000, 600]
            ],
            b=[500, 15, 1000])
        result = model.solve()
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.objective, 325)
        self.assertEqual(result.variables, ('x1', 'x2', 'x3'))
        self.assertEqual(result.values, (0.0, 2.5, 0.0))
