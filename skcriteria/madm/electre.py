#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
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

import itertools as it

import numpy as np

from ._base import KernelResult, SKCDecisionMakerABC
from ..core import Objective
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

    _skcriteria_parameters = ["p", "q"]

    def __init__(self, *, p=0.65, q=0.35):
        self._p = float(p)
        self._q = float(q)

    @property
    def p(self):
        """Concordance threshold."""
        return self._p

    @property
    def q(self):
        """Discordance threshold."""
        return self._q

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


# =============================================================================
# ELECTRE 2
# =============================================================================


# p0 (electre1), p1, p2
# q0 (electre1), q1


def weight_summatory(matrix, weights, objectives):

    alt_n = len(matrix)

    alt_combs = it.combinations(range(alt_n), 2)

    result = np.full((alt_n, alt_n), False, dtype=bool)

    for a0_idx, a1_idx in alt_combs:

        # sacamos las alternativas
        a0, a1 = matrix[[a0_idx, a1_idx]]

        # vemos donde hay maximos y donde hay minimos estrictos
        maxs, mins = (a0 > a1), (a0 < a1)

        # armamos los vectores de a \succ b teniendo en cuenta los objetivs
        a0_s_a1 = np.where(objectives == Objective.MAX.value, maxs, mins)
        a1_s_a0 = np.where(objectives == Objective.MAX.value, mins, maxs)

        # sacamos ahora los criterios
        result[a0_idx, a1_idx] = np.sum(weights * a0_s_a1) >= np.sum(
            weights * a1_s_a0
        )
        result[a1_idx, a0_idx] = np.sum(weights * a1_s_a0) >= np.sum(
            weights * a0_s_a1
        )

    return result


def electre2(
    matrix, objectives, weights, p0=0.65, p1=0.5, p2=0.35, q0=0.65, q1=0.35
):
    """Execute ELECTRE2 without any validation."""

    matrix_concordance = concordance(matrix, objectives, weights)
    matrix_discordance = discordance(matrix, objectives)
    matrix_wsum = wsum(matrix, objectives, weights)

    # creamos los grafos debiles (w) y fuertes(s)
    outrank_s = (
        (matrix_concordance >= p0) & (matrix_discordance <= q0) & matrix_wsum
    ) | ((matrix_concordance >= p1) & (matrix_discordance <= q1) & matrix_wsum)

    outrank_w = (
        (matrix_concordance >= p2) & (matrix_discordance <= q0) & matrix_wsum
    )

    len(matrix)

    import ipdb

    ipdb.set_trace()


class ELECTRE2(SKCDecisionMakerABC):
    """Find a the rankin solution through ELECTRE-2."""

    _skcriteria_parameters = ["p0", "p1", "p2", "q0", "q1"]

    def __init__(self, *, p0=0.65, p1=0.5, p2=0.35, q0=0.65, q1=0.35):
        p0, p1, p2, q0, q1 = map(float, (p0, p1, p2, q0, q1))

        if not (1 >= p0 >= p1 >= p2 >= 0):
            raise ValueError(
                "Condition '1 >= p0 >= p1 >= p2 >= 0' must be fulfilled. "
                "Found: p0={p0}, p1={p1} p2={p2}.'"
            )
        if not (1 >= q0 >= q1 >= 0):
            raise ValueError(
                "Condition '1 >= q0 >= q1 >= 0' must be fulfilled. "
                "Found: q0={q0}, q1={q1}.'"
            )

        self._p0, self._p1, self._p2, self._q0, self._q1 = (p0, p1, p2, q0, q1)

    @property
    def p0(self):
        """Concordance threshold 0."""
        return self._p0

    @property
    def p1(self):
        """Concordance threshold 1."""
        return self._p1

    @property
    def p2(self):
        """Concordance threshold 2."""
        return self._p2

    @property
    def q0(self):
        """Discordance threshold 0."""
        return self._q0

    @property
    def q1(self):
        """Discordance threshold 1."""
        return self._q1

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        (
            ranking,
            kernel,
            outrank,
            matrix_concordance,
            matrix_discordance,
        ) = electre2(
            matrix,
            objectives,
            weights,
            self.p0,
            self.p1,
            self.p2,
            self.q0,
            self.q1,
        )
        return ranking, {
            "kernel": kernel,
            "outrank": outrank,
            "matrix_concordance": matrix_concordance,
            "matrix_discordance": matrix_discordance,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return KernelResult(
            "ELECTRE1", alternatives=alternatives, values=values, extra=extra
        )
