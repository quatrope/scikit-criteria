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

from scipy import stats

from ._madm_base import KernelResult, RankResult, SKCDecisionMakerABC
from ..core import Objective
from ..utils import doc_inherit, will_change


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

    # TODO: remove loops

    kernel = ~outrank.any(axis=0)

    return kernel, outrank, matrix_concordance, matrix_discordance


class ELECTRE1(SKCDecisionMakerABC):
    """Find a kernel of alternatives through ELECTRE-1.

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
        p, q = float(p), float(q)

        if not (1 >= p >= 0):
            raise ValueError(f"p must be a value between 0 and 1. Found {p}")
        if not (1 >= q >= 0):
            raise ValueError(f"q must be a value between 0 and 1. Found {q}")

        self._p, self._q = p, q

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


def weights_outrank(matrix, weights, objectives):
    """Calculate a matrix of comparison of alternatives where the value of \
    each cell determines how many times the value of the criteria weights of \
    the row alternative exceeds those of the column alternative.

    Notes
    -----
    For more information about this matrix please check  "Tomada de decisões em
    cenários complexos" :cite:p:`gomez2004tomada`, p. 100

    """
    alt_n = len(matrix)
    alt_combs = it.combinations(range(alt_n), 2)
    outrank = np.full((alt_n, alt_n), False, dtype=bool)

    for a0_idx, a1_idx in alt_combs:

        # select the two alternatives to compare
        a0, a1 = matrix[[a0_idx, a1_idx]]

        # we see where there are strict maximums and minimums
        maxs, mins = (a0 > a1), (a0 < a1)

        # we assemble the vectors of a \succ b taking the
        # objectives into account
        a0_s_a1 = np.where(objectives == Objective.MAX.value, maxs, mins)
        a1_s_a0 = np.where(objectives == Objective.MAX.value, mins, maxs)

        # we now draw out the criteria
        outrank[a0_idx, a1_idx] = np.sum(weights * a0_s_a1) >= np.sum(
            weights * a1_s_a0
        )
        outrank[a1_idx, a0_idx] = np.sum(weights * a1_s_a0) >= np.sum(
            weights * a0_s_a1
        )

    return outrank


def _electre2_ranker(
    alt_n, original_outrank_s, original_outrank_w, invert_ranking
):

    # here we store the final rank
    ranking = np.zeros(alt_n, dtype=int)

    # copy to not destroy outrank_s and outrank_w
    outrank_s = np.copy(original_outrank_s)
    outrank_w = np.copy(original_outrank_w)

    # The alternatives still not ranked
    alt_snr_idx = np.arange(alt_n)

    # the current rank
    current_rank_position = 1

    while len(outrank_w) or len(outrank_s):

        kernel_s = ~outrank_s.any(axis=0)
        kernel_w = ~outrank_w.any(axis=0)

        # kernel strong - kernel weak
        kernel_smw = kernel_s & ~kernel_w

        # if there is no kernel, all are on equal footing and we need to assign
        # the current rank to the not evaluated alternatives.
        # After that, we can stop the loop
        if not np.any(kernel_smw):
            ranking[ranking == 0] = current_rank_position
            break

        # we create the container that will have the value of the ranking only
        # in the places to be assigned, in the other cases we leave 0
        rank_to_asign = np.zeros(alt_n, dtype=int)

        # we have to take into account that the graphs are getting smaller
        # and smaller so we need alt_snr_idx to see which alternatives still
        # need to be rank
        rank_to_asign[alt_snr_idx[kernel_smw]] = current_rank_position

        # we add the ranking to the global ranking
        # (where you do not have to add + 0)
        ranking = ranking + rank_to_asign

        # remove kernel from graphs
        to_keep = np.argwhere(~kernel_smw).flatten()

        outrank_s = outrank_s[to_keep][:, to_keep]
        outrank_w = outrank_w[to_keep][:, to_keep]
        alt_snr_idx = alt_snr_idx[to_keep]

        # next time we will assign the current ranking + 1
        current_rank_position += 1

    if invert_ranking:
        max_value = np.max(ranking)
        ranking = (max_value + 1) - ranking

    return ranking


@will_change(
    reason="electre2 implementation will change in version after 0.8",
    version=0.8,
)
def electre2(
    matrix, objectives, weights, p0=0.65, p1=0.5, p2=0.35, q0=0.65, q1=0.35
):
    """Execute ELECTRE2 without any validation."""
    matrix_concordance = concordance(matrix, objectives, weights)
    matrix_discordance = discordance(matrix, objectives)
    matrix_wor = weights_outrank(matrix, objectives, weights)

    # weak and strong graphs
    outrank_s = (
        (matrix_concordance >= p0) & (matrix_discordance <= q0) & matrix_wor
    ) | ((matrix_concordance >= p1) & (matrix_discordance <= q1) & matrix_wor)

    outrank_w = (
        (matrix_concordance >= p2) & (matrix_discordance <= q0) & matrix_wor
    )

    # number of alternatives
    alt_n = len(matrix)

    # TODO: remove loops

    # calculation of direct and indirect ranking

    ranking_direct = _electre2_ranker(
        alt_n, outrank_s, outrank_w, invert_ranking=False
    )
    ranking_inverted = _electre2_ranker(
        alt_n, outrank_s.T, outrank_w.T, invert_ranking=True
    )

    # join the two ranks
    score = (ranking_direct + ranking_inverted) / 2.0
    ranking = stats.rankdata(score, method="dense")

    return (
        ranking,
        ranking_direct,
        ranking_inverted,
        matrix_concordance,
        matrix_discordance,
        matrix_wor,
        outrank_s,
        outrank_w,
        score,
    )


@will_change(
    reason="ELECTRE2 implementation will change in version after 0.8",
    version=0.8,
)
class ELECTRE2(SKCDecisionMakerABC):
    """Find the ranking solution through ELECTRE-2.

    ELECTRE II was proposed by Roy and Bertier (1971-1973) to overcome ELECTRE
    I's inability to produce a ranking of alternatives. Instead of simply
    finding  the kernel set, ELECTRE II can order alternatives by introducing
    the strong and the weak outranking relations.

    Notes
    -----
    This implementation is based on the one presented in the book
    "Tomada de decisões em cenários complexos" :cite:p:`gomez2004tomada`.

    Parameters
    ----------
    p0, p1, p2 : float, optional (default=0.65, 0.5, 0.35)
        Matching thresholds. These are the thresholds that indicate the extent
        to which an alternative can be considered equivalent, good or very good
        with respect to another alternative.

        These thresholds must meet the condition "1 >= p0 >= p1 >= p2 >= 0".

    q0, q1 : float, optional (default=0.65, 0.35)
        Discordance threshold. Threshold of the degree to which an alternative
        is equivalent, preferred or strictly preferred to another alternative.

        These thresholds must meet the condition "1 >= q0 >= q1 >= 0".

    References
    ----------
    :cite:p:`gomez2004tomada`
    :cite:p:`roy1971methode`
    :cite:p:`roy1973methode`

    """

    _skcriteria_parameters = ["p0", "p1", "p2", "q0", "q1"]

    def __init__(self, *, p0=0.65, p1=0.5, p2=0.35, q0=0.65, q1=0.35):
        p0, p1, p2, q0, q1 = map(float, (p0, p1, p2, q0, q1))

        if not (1 >= p0 >= p1 >= p2 >= 0):
            raise ValueError(
                "Condition '1 >= p0 >= p1 >= p2 >= 0' must be fulfilled. "
                f"Found: p0={p0}, p1={p1} p2={p2}.'"
            )
        if not (1 >= q0 >= q1 >= 0):
            raise ValueError(
                "Condition '1 >= q0 >= q1 >= 0' must be fulfilled. "
                f"Found: q0={q0}, q1={q1}.'"
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
            ranking_direct,
            ranking_inverted,
            matrix_concordance,
            matrix_discordance,
            matrix_wor,
            outrank_s,
            outrank_w,
            score,
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
            "ranking_direct": ranking_direct,
            "ranking_inverted": ranking_inverted,
            "matrix_concordance": matrix_concordance,
            "matrix_discordance": matrix_discordance,
            "matrix_wor": matrix_wor,
            "outrank_s": outrank_s,
            "outrank_w": outrank_w,
            "score": score,
        }

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ELECTRE2",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
