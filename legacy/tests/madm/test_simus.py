#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017-2018, Cabral, Juan; Luczywo, Nadia
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

"""test simple methods"""


# =============================================================================
# IMPORTS
# =============================================================================

from skcriteria import Data
from ...madm import simus

from ..tcore import SKCriteriaTestCase


# =============================================================================
# Tests
# =============================================================================

class SimusTest(SKCriteriaTestCase):
    mnorm = "sum"
    wnorm = "sum"

    def setUp(self):
        # Data From:
        # Munier, N., Carignano, C., & Alberto, C.
        # UN MÉTODO DE PROGRAMACIÓN MULTIOBJETIVO.
        # Revista de la Escuela de Perfeccionamiento en Investigación
        # Operativa, 24(39).

        self.data = Data(
            mtx=[[250, 120, 20, 800],
                 [130, 200, 40, 1000],
                 [350, 340, 15, 600]],
            criteria=[max, max, min, max],
            anames=["Proyecto 1", "Proyecto 2", "Proyecto 3"],
            cnames=["Criterio 1", "Criterio 2", "Criterio 3", "Criterio 4"])
        self.b = [None, 500, None, None]

    def test_simus(self):
        dm = simus.SIMUS(njobs=1)
        dec = dm.decide(self.data, b=self.b)
        self.assertAllClose(dec.rank_, [3, 2, 1])
        self.assertAllClose(
            dec.e_.points1, [0.09090909, 0.66603535, 0.74305556])
        self.assertAllClose(
            dec.e_.points2, [-2.45454545, 0.99621211, 1.45833334])
