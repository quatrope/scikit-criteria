#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> wrote this file. As long as you retain this notice
# you can do whatever you want with this stuff. If we meet some day, and you
# think this stuff is worth it, you can buy me a WISKEY in return Juan BC


# =============================================================================
# DOCS
# =============================================================================

"""AHP

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .common import norm, rank


# =============================================================================
# CONSTANTS
# =============================================================================

#: Theorical limit of number of criteria or alternatives in AHP [SATTY2003]_
#: [SATTY2008]_
#:
#: References
#: ----------
#:  Saaty, T.L. and Ozdemir, M.S. Why the Magic Number Seven Plus or Minus Two,
#:  Mathematical and Computer Modelling, 2003, vol. 38, pp. 233-244
AHP_LIMIT = 16

MTX_TYPE_CRITERIA = "criteria"

MTX_TYPE_ALTERNATIVES = "alternatives"

SAATY_MIN, SAATY_MAX = 0, 10

#: Random indexes [SATTY1980]_
SAATY_RI = np.array(
    [(1, 0.), (2, 0.), (3, 0.58), (4, 0.90),
     (5, 1.12), (6, 1.24), (7, 1.32), (8, 1.41),
     (9, 1.45), (10, 1.49), (11, 1.51), (12, 1.48),
     (13, 1.56), (14, 1.57), (15, 1.59)],
    dtype=[('size', 'i'), ('ri', 'f2')]
)


def _resolve_saaty_intensity():
    saaty_direct = np.array([
        (1, "1", "Equal Importance",
            "Two activities contribute equally to the objective"),
        (2, "2", "Weak or slight",
            "Two activities contribute equally to the objective"),
        (3, "3", "Moderate importance",
         "Experience and judgement slightly favour one activity over another"),
        (4, "4", "Moderate plus",
         "Experience and judgement slightly favour one activity over another"),
        (5, "5", "Strong importance",
         "Experience and judgement strongly favour one activity over another"),
        (6, "6", "Strong plus",
         "Experience and judgement strongly favour one activity over another"),
        (7, "7", "Very strong or demonstrated importance", (
         "An activity is favoured very strongly over another; its "
         "dominance demonstrated in practice")),
        (8, "8", "Very, very strong", (
         "An activity is favoured very strongly over another; its "
         "dominance demonstrated in practice")),
        (9, "9", "Extreme importance",
         "The evidence favouring one activity over another")],
        dtype=[('value', 'f2'), ('label', 'a5'),
               ('definition', 'a255'), ('explanation', 'a255')]
    )

    rec_def = ("If activity i has one of the above non-zero numbers assigned "
               "to it when compared with activity j, then j has the "
               "reciprocal value when compared with i")
    rec_exp = "A reasonable assumption"

    saaty_rec = np.array([
        (1/v["value"], "1/{}".format(v["label"]), rec_def, rec_exp)
        for v in saaty_direct], dtype=saaty_direct.dtype
    )

    return np.concatenate([saaty_direct, saaty_rec])

SAATY_INTENSITY = _resolve_saaty_intensity()

del _resolve_saaty_intensity


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_values(values):
    values = np.asarray(values)
    if not np.all((values > SAATY_MIN) & (values < SAATY_MAX)):
        msg = "All values must > {} and < {}".format(SAATY_MIN, SAATY_MAX)
        raise ValueError(msg)


def validate_ahp_matrix(rows_and_columns, mtx, mtxtype=None):

    if mtxtype is not None and \
        mtxtype not in [MTX_TYPE_CRITERIA, MTX_TYPE_ALTERNATIVES]:
            msg = "'mtxtype must be 'None', '{}' or '{}'. Found '{}'".format(
                MTX_TYPE_ALTERNATIVES, MTX_TYPE_CRITERIA, mtxtype)
            raise ValueError(msg)

    if rows_and_columns > AHP_LIMIT:
        if mtxtype:
            msg = "The max number of {} must be <= {}.".format(
                mtxtype, AHP_LIMIT)
        else:
            msg = "The max number of rows and columns must be <= {}.".format(
                AHP_LIMIT)
        raise ValueError(msg)

    mtx = np.asarray(mtx)

    shape = (rows_and_columns, rows_and_columns)
    if mtx.shape != shape:
        msg = "The shape of {} matrix must be '{}'. Found '{}'".format(
            mtxtype or "the", shape, mtx.shape)
        raise ValueError(msg)

    if not np.all(np.diagonal(mtx) == 1):
        msg = "All the diagonal values must be only ones (1)"
        raise ValueError(msg)

    validate_values(mtx)

    triu, tril = np.triu(mtx), np.tril(mtx)

    # tril to triu
    old_err_state = np.seterr(divide='ignore')
    try:
        trilu = 1.0 / tril.T
    finally:
        np.seterr(**old_err_state)
    trilu[np.where(trilu == np.inf)] = 0

    if not np.allclose(triu, trilu):
        raise ValueError("The matix is not symmetric with reciprocal values")


def t(arr, dtype=float):
    shape = len(arr), len(arr[-1])

    if shape[0] != shape[1]:
        raise ValueError("The low triangular matrix for AHP must "
                         "have the same number of columns and rows")

    mtx = np.zeros(shape, dtype=dtype)

    for ridx, row in enumerate(arr):
        for cidx, value in enumerate(row):
            mtx[ridx][cidx] = value
            mtx[cidx][ridx] = 1.0 / value

    return mtx


# =============================================================================
# AHP FUNCTIONS
# =============================================================================

def saaty_closest_intensity(value):
    validate_values(value)
    deltas = np.abs(SAATY_INTENSITY["value"] - value)
    idx = np.argmin(deltas)
    return SAATY_INTENSITY[idx], deltas[idx]


def saaty_ri(size):
    sizes = SAATY_RI["size"]
    idx = np.where(sizes == size)[0][0]
    return SAATY_RI["ri"][idx]


def saaty_cr(size, mtx):
    validate_ahp_matrix(size, mtx)
    colsum = np.sum(mtx, axis=0)
    nmtx = np.divide(mtx, colsum, dtype="f")
    avg = np.average(nmtx, axis=1)
    lambda_max = np.dot(colsum, avg)
    ci = (lambda_max - size) / (size - 1)
    ri = saaty_ri(size)
    return ci, ci/ri


def ahp(crit_n, alt_n, crit_vs_crit, alt_vs_alt_by_crit):
    """ """

    # criteria
    validate_ahp_matrix(crit_n, crit_vs_crit, mtxtype=MTX_TYPE_CRITERIA)
    n_cvsc = norm.colsum(crit_vs_crit, axis=0)
    pvector = np.average(n_cvsc, axis=1)

    # alternatives
    if len(alt_vs_alt_by_crit) != crit_n:
        msg = (
            "The number 'alt_vs_alt_by_crit' must be "
            "the number of criteria '{}'. Found"
        ).format(crit_n, len(alt_vs_alt_by_crit))
        raise ValueError(msg)

    pmatrix = np.empty((crit_n, alt_n))
    for cidx, altmtx in enumerate(alt_vs_alt_by_crit):
        validate_ahp_matrix(alt_n, altmtx, mtxtype=MTX_TYPE_ALTERNATIVES)
        n_altmtx = norm.colsum(altmtx, axis=0)
        pmatrix[:, cidx] = np.average(n_altmtx, axis=1)

    points = np.dot(pmatrix, pvector)

    return rank.rankdata(points, reverse=True), points


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
