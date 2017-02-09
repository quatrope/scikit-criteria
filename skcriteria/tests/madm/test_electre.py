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

__doc__ = """test electre methods"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import core
from ... import util, norm
from ...madm import electre


# =============================================================================
# BASE CLASS
# =============================================================================

class ConcordanceTest(core.SKCriteriaTestCase):

    def test_concordance(self):
        # Data From:
        # Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        # en Aragón: Una aplicación empírica de la ayuda a la decisión
        # multicriterio tipo ELECTRE I y III. Robustez de los resultados
        # obtenidos.
        # Revista de Métodos Cuantitativos para la Economía y la Empresa,
        # (7), 31-56.

        nmtx = norm.sum([
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8]
        ], axis=0)
        ncriteria = util.criteriarr([1, 1, -1, 1, 1])
        nweights = norm.sum([0.25, 0.25, 0.1, 0.2, 0.2])
        results = [
            [np.nan, 0.5000, 0.3500, 0.5000, 0.3500, 0.4500],
            [0.5000, np.nan, 0.5000, 0.7500, 0.5000, 0.5000],
            [0.6500, 0.5000, np.nan, 0.4500, 0.2000, 0.7000],
            [0.7500, 0.2500, 0.5500, np.nan, 0.3500, 0.4500],
            [0.9000, 0.7000, 0.8000, 0.9000, np.nan, 0.9000],
            [0.5500, 0.5000, 0.5500, 0.5500, 0.1000, np.nan]
        ]
        concordance = electre.concordance(nmtx, ncriteria, nweights)
        self.assertAllClose(concordance, results, atol=1.e-3)


class DiscordanceTest(core.SKCriteriaTestCase):

    def test_discordance(self):
        # Data From:
        # Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        # en Aragón: Una aplicación empírica de la ayuda a la decisión
        # multicriterio tipo ELECTRE I y III. Robustez de los resultados
        # obtenidos.
        # Revista de Métodos Cuantitativos para la Economía y la Empresa,
        # (7), 31-56.

        nmtx = norm.sum([
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8]
        ], axis=0)
        ncriteria = util.criteriarr([1, 1, -1, 1, 1])
        results = [
            [np.nan, 1.0000, 0.6667, 0.5000, 1.0000, 0.7500],
            [1.0000, np.nan, 0.7143, 1.0000, 1.0000, 0.5714],
            [0.7000, 1.0000, np.nan, 0.8000, 0.7500, 0.9000],
            [0.5714, 0.6667, 0.8571, np.nan, 1.0000, 0.7143],
            [0.2000, 0.5000, 0.3333, 0.3000, np.nan, 0.4000],
            [0.5000, 1.0000, 0.8333, 0.5000, 0.5000, np.nan]
        ]

        discordance = electre.discordance(nmtx, ncriteria)
        self.assertAllClose(discordance, results, atol=1.e-3)


class Electre1Test(core.SKCriteriaTestCase):
    mnorm = "sum"
    wnorm = "sum"

    def test_electre1(self):
        # Data From:
        # Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        # en Aragón: Una aplicación empírica de la ayuda a la decisión
        # multicriterio tipo ELECTRE I y III. Robustez de los resultados
        # obtenidos.
        # Revista de Métodos Cuantitativos para la Economía y la Empresa,
        # (7), 31-56.

        mtx = [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8]
        ]
        criteria = [1, 1, -1, 1, 1]
        weights = [0.25, 0.25, 0.1, 0.2, 0.2]

        result_kernel = [4]
        result_outrank = [
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [True, False, False, False, False, False],
            [True, True, True, True, False, True],
            [False, False, False, False, False, False]
        ]
        result_concordance = [
            [np.nan, 0.5000, 0.3500, 0.5000, 0.3500, 0.4500],
            [0.5000, np.nan, 0.5000, 0.7500, 0.5000, 0.5000],
            [0.6500, 0.5000, np.nan, 0.4500, 0.2000, 0.7000],
            [0.7500, 0.2500, 0.5500, np.nan, 0.3500, 0.4500],
            [0.9000, 0.7000, 0.8000, 0.9000, np.nan, 0.9000],
            [0.5500, 0.5000, 0.5500, 0.5500, 0.1000, np.nan]
        ]
        result_discordance = [
            [np.nan, 1.0000, 0.6667, 0.5000, 1.0000, 0.7500],
            [1.0000, np.nan, 0.7143, 1.0000, 1.0000, 0.5714],
            [0.7000, 1.0000, np.nan, 0.8000, 0.7500, 0.9000],
            [0.5714, 0.6667, 0.8571, np.nan, 1.0000, 0.7143],
            [0.2000, 0.5000, 0.3333, 0.3000, np.nan, 0.4000],
            [0.5000, 1.0000, 0.8333, 0.5000, 0.5000, np.nan]
        ]

        nmtx, ncriteria, nweights = self.normalize(mtx, criteria, weights)
        kernel, outrank, concordance, discordance = electre.electre1(
            nmtx, ncriteria, nweights=nweights, p=0.5500, q=0.699)

        self.assertCountEqual(kernel, result_kernel)
        self.assertArrayEqual(outrank, result_outrank)
        self.assertAllClose(concordance, result_concordance, atol=1.e-3)
        self.assertAllClose(discordance, result_discordance, atol=1.e-3)

    def test_electre1_dm(self):
        # Data From:
        # Cebrián, L. I. G., & Porcar, A. M. (2009). Localización empresarial
        # en Aragón: Una aplicación empírica de la ayuda a la decisión
        # multicriterio tipo ELECTRE I y III. Robustez de los resultados
        # obtenidos.
        # Revista de Métodos Cuantitativos para la Economía y la Empresa,
        # (7), 31-56.

        mtx = [
            [6, 5, 28, 5, 5],
            [4, 2, 25, 10, 9],
            [5, 7, 35, 9, 6],
            [6, 1, 27, 6, 7],
            [6, 8, 30, 7, 9],
            [5, 6, 26, 4, 8]
        ]
        criteria = [1, 1, -1, 1, 1]
        weights = [0.25, 0.25, 0.1, 0.2, 0.2]

        result_kernel = [4]
        result_outrank = [
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [False, False, False, False, False, False],
            [True, False, False, False, False, False],
            [True, True, True, True, False, True],
            [False, False, False, False, False, False]
        ]
        result_concordance = [
            [np.nan, 0.5000, 0.3500, 0.5000, 0.3500, 0.4500],
            [0.5000, np.nan, 0.5000, 0.7500, 0.5000, 0.5000],
            [0.6500, 0.5000, np.nan, 0.4500, 0.2000, 0.7000],
            [0.7500, 0.2500, 0.5500, np.nan, 0.3500, 0.4500],
            [0.9000, 0.7000, 0.8000, 0.9000, np.nan, 0.9000],
            [0.5500, 0.5000, 0.5500, 0.5500, 0.1000, np.nan]
        ]
        result_discordance = [
            [np.nan, 1.0000, 0.6667, 0.5000, 1.0000, 0.7500],
            [1.0000, np.nan, 0.7143, 1.0000, 1.0000, 0.5714],
            [0.7000, 1.0000, np.nan, 0.8000, 0.7500, 0.9000],
            [0.5714, 0.6667, 0.8571, np.nan, 1.0000, 0.7143],
            [0.2000, 0.5000, 0.3333, 0.3000, np.nan, 0.4000],
            [0.5000, 1.0000, 0.8333, 0.5000, 0.5000, np.nan]
        ]

        dm = electre.ELECTRE1(p=0.55, q=0.699)
        decision = dm.decide(mtx, criteria, weights)

        self.assertCountEqual(decision.kernel_, result_kernel)
        self.assertArrayEqual(decision.e_.outrank, result_outrank)
        self.assertAllClose(
            decision.e_.mtx_concordance, result_concordance, atol=1.e-3)
        self.assertAllClose(
            decision.e_.mtx_discordance, result_discordance, atol=1.e-3)
