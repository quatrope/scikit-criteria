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

from skcriteria.common import norm, util, rank


# =============================================================================
# UTILS
# =============================================================================

def concordance(nmtx, ncriteria, nweights):

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


def discordance(nmtx, ncriteria, nweights):

    mtx_criteria = np.tile(ncriteria, (len(nmtx), 1))
    mtx_weight = np.tile(nweights, (len(nmtx), 1))
    mtx_discordance = np.empty((len(nmtx), len(nmtx)))
    ranges = np.max(nmtx, axis=0) - np.min(nmtx, axis=0)

    for idx, row in enumerate(nmtx):
        difference = nmtx - row
        worsts = (
            ((mtx_criteria == util.MAX) & (difference >= 0)) |
            ((mtx_criteria == util.MIN) & (difference <= 0))
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

def electre1(mtx, criteria, weights=1):

    # This guarantee the criteria array consistency
    ncriteria = util.criteriarr(criteria)

    # validate the matrix is the matrix
    nmtx = np.asarray(mtx)
    if not util.is_mtx(nmtx):
        raise ValueError("'mtx' is not a matrix")

    # normalize weights
    nweights = norm.sum(weights) if weights is not None else 1

    # get the concordance matrix
    mtx_concordance = concordance(nmtx, nweights)

