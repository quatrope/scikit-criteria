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

import numpy as np

import joblib

from ...validate import criteriarr
from ...base import Data
from ... import norm

from ...madm import electre

from ..tcore import SKCriteriaTestCase


# =============================================================================
# BASE CLASS
# =============================================================================

class ConcordanceTest(SKCriteriaTestCase):

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
        ncriteria = criteriarr([1, 1, -1, 1, 1])
        nweights = norm.sum([0.25, 0.25, 0.1, 0.2, 0.2])
        results = [
            [np.nan, 0.5000, 0.3500, 0.5000, 0.3500, 0.4500],
            [0.5000, np.nan, 0.5000, 0.7500, 0.5000, 0.5000],
            [0.6500, 0.5000, np.nan, 0.4500, 0.2000, 0.7000],
            [0.7500, 0.2500, 0.5500, np.nan, 0.3500, 0.4500],
            [0.9000, 0.7000, 0.8000, 0.9000, np.nan, 0.9000],
            [0.5500, 0.5000, 0.5500, 0.5500, 0.1000, np.nan]
        ]
        with joblib.Parallel(n_jobs=1) as jobs:
            concordance = electre.concordance(nmtx, ncriteria, nweights, jobs)
        self.assertAllClose(concordance, results, atol=1.e-3)


class DiscordanceTest(SKCriteriaTestCase):

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
        ncriteria = criteriarr([1, 1, -1, 1, 1])
        results = [
            [np.nan, 0.5052, 0.4042, 0.1883, 0.42857, 0.2825],
            [0.4286, np.nan, 0.7143, 0.2589, 0.8571, 0.5714],
            [0.1696, 0.2825, np.nan, 0.1938, 0.2825, 0.2180],
            [0.5714, 0.4042, 0.8571, np.nan, 1.0, 0.7143],
            [0.0485, 0.3031, 0.2021, 0.0727, np.nan, 0.0969],
            [0.1295, 0.6063, 0.5052, 0.2021, 0.3031, np.nan]]

        with joblib.Parallel(n_jobs=1) as jobs:
            discordance = electre.discordance(nmtx, ncriteria, jobs)
        self.assertAllClose(discordance, results, atol=1.e-3)


class Electre1Test(SKCriteriaTestCase):
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
        dm = electre.ELECTRE1(p=0.55, q=0.699, njobs=1)
        decision = dm.decide(mtx, criteria, weights)
        self.assertCountEqual(decision.kernel_, result_kernel)

    def test_kernel_sensibility(self):
        # Barba-Romero, S., & Pomerol, J. C. (1997).
        # Decisiones multicriterio: fundamentos teóricos y utilización
        # práctica. P.216
        mtx = [
            [0.188, 0.172, 0.168, 0.122, 0.114],
            [0.125, 0.069, 0.188, 0.244, 0.205],
            [0.156, 0.241, 0.134, 0.220, 0.136],
            [0.188, 0.034, 0.174, 0.146, 0.159],
            [0.188, 0.276, 0.156, 0.171, 0.205],
            [0.156, 0.207, 0.180, 0.098, 0.182]]
        weights = [0.25, 0.25, 0.10, 0.20, 0.20]
        criteria = [1, 1, 1, 1, 1]
        anames = ["A", "B", "D", "E", "G", "H"]

        data = Data(mtx, criteria, weights, anames)
        ps = [0.50, 0.60, 0.70, 0.80, 0.89, 0.89, 0.89, 0.94, 1]
        qs = [0.50, 0.40, 0.30, 0.20, 0.10, 0.08, 0.05, 0.05, 0]

        kernels_len = []
        for p, q in zip(ps, qs):
            dm = electre.ELECTRE1(
                mnorm="none", wnorm="none", p=p, q=q, njobs=1)
            dec = dm.decide(data)
            klen = len(dec.kernel_)

            if kernels_len and klen < kernels_len[-1]:
                self.fail("less sensitive electre must have more "
                          "alternatives in kernel. {}".format(kernels_len))
            kernels_len.append(klen)

    @property
    def njobs(self):
        return self._njobs
