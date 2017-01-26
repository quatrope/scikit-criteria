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

def concordance(nmtx, ncriteria, nweights=1):

    mtx_criteria = np.tile(ncriteria, (len(nmtx), 1))
    mtx_weight = np.tile(nweights, (len(nmtx), 1))
    mtx_concordance = np.empty((len(nmtx), len(nmtx)))

    for idx, row in enumerate(nmtx):
        difference = row - nmtx
        outrank = (
            ((mtx_criteria == util.MAX) & (difference >= 0)) |
            ((mtx_criteria == util.MIN) & (difference <= 0))
        )
        filter_weights = mtx_weight * outrank.astype(int)
        new_row = np.sum(filter_weights, axis=1)
        mtx_concordance[idx] = new_row

    np.fill_diagonal(mtx_concordance, np.nan)
    mean = np.nanmean(mtx_concordance)
    p = util.nearest(mtx_concordance, mean, side="gt")

    return mtx_concordance, mean, p


def discordance(nmtx, ncriteria):

    mtx_criteria = np.tile(ncriteria, (len(nmtx), 1))
    mtx_discordance = np.empty((len(nmtx), len(nmtx)))
    ranges = np.max(nmtx, axis=0) - np.min(nmtx, axis=0)

    for idx, row in enumerate(nmtx):
        difference = nmtx - row
        worsts = (
            ((mtx_criteria == util.MAX) & (difference > 0)) |
            ((mtx_criteria == util.MIN) & (difference < 0))
        )
        filter_difference = np.abs(difference * worsts)
        delta = filter_difference / ranges
        new_row = np.max(delta, axis=1)
        mtx_discordance[idx] = new_row

    np.fill_diagonal(mtx_discordance, np.nan)
    mean = np.nanmean(mtx_discordance)
    q = util.nearest(mtx_discordance, mean, side="lt")

    return mtx_discordance, mean, q


# =============================================================================
# ELECTRE
# =============================================================================

def electre1(mtx, criteria, weights=None):

    # This guarantee the criteria array consistency
    ncriteria = util.criteriarr(criteria)

    # normalize
    nmtx = norm.sum(mtx)
    nweights = norm.sum(weights) if weights is not None else 1

    # get the concordance and discordance info
    mtx_concordance, _, p = concordance(nmtx, ncriteria, nweights)
    mtx_discordance, _, q = discordance(nmtx, ncriteria)

    with np.errstate(invalid='ignore'):
        outrank = (
            (mtx_concordance > p) & (mtx_discordance < q))

    bt_rows = np.sum(outrank, axis=1)
    bt_columns = np.sum(outrank, axis=0)

    diff = bt_rows - bt_columns
    max_value = np.max(diff)

    kernel = np.where((diff == max_value) & (diff > 0))[0]

    return kernel, outrank, mtx_concordance, mtx_discordance, p, q


# =============================================================================
# OO
# =============================================================================

class ELECTRE1(DecisionMaker):

    def solve(self, *args, **kwargs):
        kernel, outrank, mtx_concordance, mtx_discordance, p, q = electre1(
            *args, **kwargs)

        extra = {
            "outrank": outrank,
            "mtx_concordance": mtx_concordance,
            "mtx_discordance": mtx_discordance,
            "p": p, "q": q}

        return kernel, None, extra
