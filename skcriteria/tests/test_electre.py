#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


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

import random

import numpy as np

from . import core

from ..common import norm, util
from .. import electre


# =============================================================================
# BASE CLASS
# =============================================================================

class ElectreTest(core.SKCriteriaTestCase):

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
        result_mean, result_p = 0.5400, 0.5500
        concordance, mean, p = electre.concordance(nmtx, ncriteria, nweights)
        self.assertAllClose(concordance, results, atol=1.e-3)
        self.assertAllClose(mean, result_mean, atol=1.e-3)
        self.assertAllClose(p, result_p, atol=1.e-3)

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
        nweights = norm.sum([0.25, 0.25, 0.1, 0.2, 0.2])
        results = [
            [np.nan, 1.0000, 0.6667, 0.5000, 1.0000, 0.7500],
            [1.0000, np.nan, 0.7143, 1.0000, 1.0000, 0.5714],
            [0.7000, 1.0000, np.nan, 0.8000, 0.7500, 0.9000],
            [0.5714, 0.6667, 0.8571, np.nan, 1.0000, 0.7143],
            [0.2000, 0.5000, 0.3333, 0.3000, np.nan, 0.4000],
            [0.5000, 1.0000, 0.8333, 0.5000, 0.5000, np.nan]
        ]
        result_mean, result_p = 0.7076, 0.70
        discordance, mean, p = electre.discordance(nmtx, ncriteria, nweights)
        self.assertAllClose(discordance, results, atol=1.e-3)
        self.assertAllClose(mean, result_mean, atol=1.e-3)
        self.assertAllClose(p, result_p, atol=1.e-3)

    def _test_electre1(self):
        #~ Data from:
        #~ Tzeng, G. H., & Huang, J. J. (2011). Multiple
        #~ attribute decision making: methods and applications. CRC press.
        VD, D, U, S, VS = 1, 2, 3, 4, 5
        mtx = [
            [U, D, S, VS],
            [D, VS, D, S],
            [D, VD, S, D],
            [U, VS, U, S]
        ]
        criteria = [1, 1, 1, 1]
        weights = [.35, .15, .20, .30]
        electre.electre1(mtx, criteria, weights)


