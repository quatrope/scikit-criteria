# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..base import SKCDecisionMakerMixin
from ..data import Objective, KernelResult
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


class ELECTRE1(SKCDecisionMakerMixin):
    def __init__(self, p=0.65, q=0.35):
        self.p = p
        self.q = q

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = float(p)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = float(q)

    @doc_inherit(SKCDecisionMakerMixin._validate_data)
    def _validate_data(self, objectives, **kwargs):
        ...

    @doc_inherit(SKCDecisionMakerMixin._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):
        kernel, outrank, matrix_concordance, matrix_discordance = electre1(
            matrix, objectives, weights, self._p, self._q
        )
        return kernel, {
            "outrank": outrank,
            "matrix_concordance": matrix_concordance,
            "matrix_discordance": matrix_discordance,
        }

    @doc_inherit(SKCDecisionMakerMixin._make_result)
    def _make_result(self, anames, values, extra):
        return KernelResult(
            "ELECTRE1", anames=anames, values=values, extra=extra
        )
