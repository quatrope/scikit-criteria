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

__doc__ = """Test io functionalities"""


# =============================================================================
# IMPORTS
# =============================================================================

import random

from six import StringIO

import numpy as np

from .. import io, dmaker

from . import core, utils


# =============================================================================
# BASE
# =============================================================================

class DMIOTest(core.SKCriteriaTestCase):

    def setUp(self):
        # if som test import a module with a decision maker
        # this function will find it
        self.dmakers = utils.collect_subclasses(dmaker.DecisionMaker)

    def test_dumps_loads(self):
        for dmcls in self.dmakers:
            dm = dmcls()
            dumped = io.dumps(dm)
            result = io.loads(dumped)
            self.assertEquals(result, dm)

            result = io.loads(dumped, skcm_metadata=True)
            self.assertEquals(result, io.jt.loads(dumped))

    def test_dump_load(self):
        for dmcls in self.dmakers:
            dm, fp = dmcls(), StringIO()
            io.dump(dm, fp)
            fp.seek(0)
            result = io.load(fp)
            self.assertEquals(result, dm)

            fp.seek(0)
            result = io.load(fp, skcm_metadata=True)
            fp.seek(0)
            self.assertEquals(result, io.jt.load(fp))

    def test_dump_loads(self):
        for dmcls in self.dmakers:
            dm, fp = dmcls(), StringIO()
            io.dump(dm, fp)
            result = io.loads(fp.getvalue())
            self.assertEquals(result, dm)

            result = io.loads(fp.getvalue(), skcm_metadata=True)
            self.assertEquals(result, io.jt.loads(fp.getvalue()))

    def test_dumps_load(self):
        for dmcls in self.dmakers:
            dm = dmcls()
            dumped = io.dumps(dm)
            fp = StringIO(dumped)
            result = io.load(fp)
            self.assertEquals(result, dm)

            fp.seek(0)
            result = io.load(fp, skcm_metadata=True)
            fp.seek(0)
            self.assertEquals(result, io.jt.load(fp))


class DecisionIOTest(core.SKCriteriaTestCase):

    def setUp(self):
        # if som test import a module with a decision maker
        # this function will find it
        self.dmakers = utils.collect_subclasses(dmaker.DecisionMaker)
        self.mtx = np.random.rand(3, 10)
        self.criteria = np.asarray([random.choice((1, -1)) for e in range(10)])
        self.weights = np.random.randint(1, 100, 10)

    def test_dumps_loads(self):
        for dmcls in self.dmakers:
            dm = dmcls()
            dec = dm.decide(self.mtx, self.criteria, self.weights)

            dumped = io.dumps(dec)
            result = io.loads(dumped)
            self.assertEquals(result, dec)

            result = io.loads(dumped, skcm_metadata=True)
            self.assertEquals(result, io.jt.loads(dumped))

    def test_dump_load(self):
        for dmcls in self.dmakers:
            dm, fp = dmcls(), StringIO()
            dec = dm.decide(self.mtx, self.criteria, self.weights)

            io.dump(dec, fp)
            fp.seek(0)
            result = io.load(fp)
            self.assertEquals(result, dec)

            fp.seek(0)
            result = io.load(fp, skcm_metadata=True)
            fp.seek(0)
            self.assertEquals(result, io.jt.load(fp))

    def test_dump_loads(self):
        for dmcls in self.dmakers:
            dm, fp = dmcls(), StringIO()
            dec = dm.decide(self.mtx, self.criteria, self.weights)

            io.dump(dec, fp)
            result = io.loads(fp.getvalue())
            self.assertEquals(result, dec)

            result = io.loads(fp.getvalue(), skcm_metadata=True)
            self.assertEquals(result, io.jt.loads(fp.getvalue()))

    def test_dumps_load(self):
        for dmcls in self.dmakers:
            dm = dmcls()
            dec = dm.decide(self.mtx, self.criteria, self.weights)
            dumped = io.dumps(dec)
            fp = StringIO(dumped)
            result = io.load(fp)
            self.assertEquals(result, dec)

            fp.seek(0)
            result = io.load(fp, skcm_metadata=True)
            fp.seek(0)
            self.assertEquals(result, io.jt.load(fp))
