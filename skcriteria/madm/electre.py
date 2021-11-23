#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""ELimination Et Choix Traduisant la REalité - ELECTRE.

ELECTRE is a family of multi-criteria decision analysis methods
that originated in Europe in the mid-1960s. The acronym ELECTRE stands for:
ELimination Et Choix Traduisant la REalité (ELimination and Choice Expressing
REality).

Usually the ELECTRE Methods are used to discard some alternatives to the
problem, which are unacceptable. After that we can use another MCDA to select
the best one. The Advantage of using the Electre Methods before is that we
can apply another MCDA with a restricted set of alternatives saving much time.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..core import KernelResult, Objective, SKCDecisionMakerABC
from ..utils import doc_inherit

# =============================================================================
# CONCORDANCE
# =============================================================================


def _conc_row(row, matrix, matrix_objectives, matrix_weights):
    difference = row - matrix
    outrank = (
        (matrix_objectives == Objective.MAX.value) & (difference >= 0)
    ) | ((matrix_objectives == Objective.MIN.value) & (difference <= 0))
    filter_weights = matrix_weights * outrank.astype(int)
    new_row = np.sum(filter_weights, axis=1)
    return new_row


def concordance(matrix, objectives, weights):
    """Calculate the concordance matrix."""
    matrix_len = len(matrix)

    matrix_objectives = np.tile(objectives, (matrix_len, 1))
    matrix_weights = np.tile(weights, (matrix_len, 1))
    matrix_concordance = np.empty((matrix_len, matrix_len), dtype=float)

    for idx, row in enumerate(matrix):
        new_row = _conc_row(row, matrix, matrix_objectives, matrix_weights)
        matrix_concordance[idx] = new_row

    np.fill_diagonal(matrix_concordance, np.nan)
    return matrix_concordance


# =============================================================================
# DISCORDANCE
# =============================================================================


def _disc_row(row, mtx, matrix_objectives, max_range):
    difference = mtx - row
    worsts = (
        (matrix_objectives == Objective.MAX.value) & (difference > 0)
    ) | ((matrix_objectives == Objective.MIN.value) & (difference < 0))
    filter_difference = np.abs(difference * worsts)
    delta = filter_difference / max_range
    new_row = np.max(delta, axis=1)
    return new_row


def discordance(matrix, objectives):
    """Calculate the discordance matrix."""
    matrix_len = len(matrix)

    matrix_objectives = np.tile(objectives, (matrix_len, 1))
    max_range = (np.max(matrix, axis=0) - np.min(matrix, axis=0)).max()
    matrix_discordance = np.empty((matrix_len, matrix_len), dtype=float)

    for idx, row in enumerate(matrix):
        matrix_discordance[idx] = _disc_row(
            row, matrix, matrix_objectives, max_range
        )

    np.fill_diagonal(matrix_discordance, np.nan)
    return matrix_discordance


# =============================================================================
# ELECTRE 1
# =============================================================================


def electre1(matrix, objectives, weights, p=0.65, q=0.35):
    """Execute ELECTRE1 without any validation."""
    # get the concordance and discordance info
    matrix_concordance = concordance(matrix, objectives, weights)
    matrix_discordance = discordance(matrix, objectives)

    with np.errstate(invalid="ignore"):
        outrank = (matrix_concordance >= p) & (matrix_discordance <= q)

    kernel = ~outrank.any(axis=0)

    return kernel, outrank, matrix_concordance, matrix_discordance


class ELECTRE1(SKCDecisionMakerABC):
    """Find a the kernel solution through ELECTRE-1.

    The ELECTRE I model find the kernel solution in a situation where true
    criteria and restricted outranking relations are given.

    That is, ELECTRE I cannot derive the ranking of alternatives but the kernel
    set. In ELECTRE I, two indices called the concordance index and the
    discordance index are used to measure the relations between objects


    Parameters
    ----------
    p : float, optional (default=0.65)
        Concordance threshold. Threshold of how much one alternative is at
        least as good as another to be significative.

    q : float, optional (default=0.35)
        Discordance threshold. Threshold of how much the degree one alternative
        is strictly preferred to another to be significative.

    References
    ----------
    :cite:p:`roy1990outranking`
    :cite:p:`roy1968classement`
    :cite:p:`tzeng2011multiple`

    """

    def __init__(self, p=0.65, q=0.35):
        self.p = p
        self.q = q

    @property
    def p(self):
        """Concordance threshold."""
        return self._p

    @p.setter
    def p(self, p):
        self._p = float(p)

    @property
    def q(self):
        """Discordance threshold."""
        return self._q

    @q.setter
    def q(self, q):
        self._q = float(q)

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        kernel, outrank, matrix_concordance, matrix_discordance = electre1(
            matrix, objectives, weights, self.p, self.q
        )
        return kernel, {
            "outrank": outrank,
            "matrix_concordance": matrix_concordance,
            "matrix_discordance": matrix_discordance,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return KernelResult(
            "ELECTRE1", alternatives=alternatives, values=values, extra=extra
        )
