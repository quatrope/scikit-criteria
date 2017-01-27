#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


__doc__ = """

Data from Tzeng and Huang et al, 2011 [TZENG2011]_

References
----------

.. [TZENG2011] Tzeng, G. H., & Huang, J. J. (2011). Multiple
   attribute decision making: methods and applications. CRC press.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import norm, util
from ..dmaker import DecisionMaker


# =============================================================================
# UTILS
# =============================================================================

def concordance(mtx, criteria, weights):

    mtx_criteria = np.tile(criteria, (len(mtx), 1))
    mtx_weight = np.tile(weights, (len(mtx), 1))
    mtx_concordance = np.empty((len(mtx), len(mtx)))

    for idx, row in enumerate(mtx):
        difference = row - mtx
        outrank = (
            ((mtx_criteria == util.MAX) & (difference >= 0)) |
            ((mtx_criteria == util.MIN) & (difference <= 0))
        )
        filter_weights = mtx_weight * outrank.astype(int)
        new_row = np.sum(filter_weights, axis=1)
        mtx_concordance[idx] = new_row

    np.fill_diagonal(mtx_concordance, np.nan)
    return mtx_concordance


def discordance(mtx, criteria):

    mtx_criteria = np.tile(criteria, (len(mtx), 1))
    mtx_discordance = np.empty((len(mtx), len(mtx)))
    ranges = np.max(mtx, axis=0) - np.min(mtx, axis=0)

    for idx, row in enumerate(mtx):
        difference = mtx - row
        worsts = (
            ((mtx_criteria == util.MAX) & (difference > 0)) |
            ((mtx_criteria == util.MIN) & (difference < 0))
        )
        filter_difference = np.abs(difference * worsts)
        delta = filter_difference / ranges
        new_row = np.max(delta, axis=1)
        mtx_discordance[idx] = new_row

    np.fill_diagonal(mtx_discordance, np.nan)
    return mtx_discordance


# =============================================================================
# ELECTRE
# =============================================================================

def electre1(mtx, criteria, p, q, weights=None):

    # This guarantee the criteria array consistency
    ncriteria = util.criteriarr(criteria)

    # normalize
    nmtx = norm.sum(mtx)
    nweights = norm.sum(weights) if weights is not None else 1

    # get the concordance and discordance info
    mtx_concordance = concordance(nmtx, ncriteria, nweights)
    mtx_discordance = discordance(nmtx, ncriteria)

    with np.errstate(invalid='ignore'):
        outrank = (
            (mtx_concordance > p) & (mtx_discordance < q))

    bt_rows = np.sum(outrank, axis=1)
    bt_columns = np.sum(outrank, axis=0)

    diff = bt_rows - bt_columns
    max_value = np.max(diff)

    kernel = np.where((diff == max_value) & (diff > 0))[0]

    return kernel, outrank, mtx_concordance, mtx_discordance


# =============================================================================
# OO
# =============================================================================

class ELECTRE1(DecisionMaker):

    def __init__(self, p=.65, q=.35, *args, **kwargs):
        super(ELECTRE1, self).__init__(*args, **kwargs)
        self.p = p
        self.q = q

    def solve(self, mtx, criteria, weights=None):
        kernel, outrank, mtx_concordance, mtx_discordance = electre1(
            mtx=mtx, criteria=criteria, weights=weights, p=self.p, q=self.q)

        extra = {
            "outrank": outrank,
            "mtx_concordance": mtx_concordance,
            "mtx_discordance": mtx_discordance,
            "p": self.p, "q": self.q}

        return kernel, None, extra
