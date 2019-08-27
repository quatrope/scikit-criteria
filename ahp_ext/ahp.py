#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


# =============================================================================
# DOCS
# =============================================================================

"""AHP"""


# =============================================================================
# IMPORTS
# =============================================================================

from collections import namedtuple

import numpy as np

from skcriteria import norm, rank


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

#: Random indexes [CHANGSHENG2013]_
SAATY_RI = {
    k: np.float16(v) for k, v in {
        1: 0.0,
        2: 0.0,
        3: 0.52,
        4: 0.89,
        5: 1.12,
        6: 1.26,
        7: 1.36,
        8: 1.41,
        9: 1.46,
        10: 1.49,
        11: 1.52,
        12: 1.54,
        13: 1.56,
        14: 1.58,
        15: 1.59
    }.items()
}


def _resolve_saaty_intensity():
    Intensity = namedtuple(
        "Intensity", ["value", "label", "definition", "explanation"])
    saaty_direct = (
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
         "The evidence favouring one activity over another"),
    )

    rec_defn = ("If activity i has one of the above non-zero numbers assigned "
                "to it when compared with activity j, then j has the "
                "reciprocal value when compared with i")
    rec_expl = "A reasonable assumption"

    saaty_intensity = {}
    for value, label, defn, expl in saaty_direct:
        saaty_intensity[value] = Intensity(value, label, defn, expl)
        rec_value = 1/float(value)
        rec_label = "1/{}".format(label)
        saaty_intensity[rec_value] = Intensity(
            rec_value, rec_label, rec_defn, rec_expl)
    return saaty_intensity

SAATY_INTENSITY = _resolve_saaty_intensity()

SAATY_INTENSITY_VALUES = np.array(list(SAATY_INTENSITY.keys()))

del _resolve_saaty_intensity


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_values(values):
    values = np.asarray(values)
    if not np.all((values > SAATY_MIN) & (values < SAATY_MAX)):
        msg = "All values must >= {} and <= {}"
        raise ValueError(msg.format(SAATY_MIN+1, SAATY_MAX-1))


def validate_ahp_matrix(rows_and_columns, mtx, mtxtype=None):
    type_validation = mtxtype is None or (
       isinstance(mtxtype, str) and
       mtxtype in [MTX_TYPE_CRITERIA, MTX_TYPE_ALTERNATIVES])

    if not type_validation:
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


def t(arr, dtype=np.float64):
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
    deltas = np.abs(SAATY_INTENSITY_VALUES - value)
    idx = np.argmin(deltas)
    closest = SAATY_INTENSITY_VALUES[idx]
    return SAATY_INTENSITY[closest]


def saaty_ri(size):
    return SAATY_RI[size]


def saaty_cr(mtx):
    size = len(mtx)
    nmtx = norm.sum(mtx, axis=0)
    weights = np.average(nmtx, axis=1)
    lambda_max = np.sum(np.dot(mtx, weights) / weights) / size
    ci = (lambda_max - size) / (size - 1)
    cr = ci / saaty_ri(size)
    return ci, cr, weights


def ahp(crit_vs_crit, alt_vs_alt_by_crit):
    """ """
    crit_n = len(crit_vs_crit)

    if len(alt_vs_alt_by_crit) != crit_n:
        msg = (
            "The number 'alt_vs_alt_by_crit' must be "
            "the number of criteria '{}'. Found"
        ).format(crit_n, len(alt_vs_alt_by_crit))
        raise ValueError(msg)

    # criteria
    crit_ci, crit_cr, wvector = saaty_cr(crit_vs_crit)
    alt_n = len(alt_vs_alt_by_crit[0])

    wmatrix = np.empty((crit_n, alt_n))
    avabc_ci, avabc_cr = np.empty(crit_n), np.empty(crit_n)
    for cidx, altmtx in enumerate(alt_vs_alt_by_crit):
        ava_ci, ava_cr, ava_weights = saaty_cr(altmtx)
        avabc_ci[cidx], avabc_cr[cidx] = ava_ci, ava_cr
        wmatrix[:, cidx] = ava_weights

    points = np.dot(wmatrix, wvector)
    ranked = rank.rankdata(points, reverse=True)

    return ranked, points, crit_ci, avabc_ci, crit_cr, avabc_cr
