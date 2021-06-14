#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.data

"""


# =============================================================================
# IMPORTS
# =============================================================================

import string

import numpy as np

import pytest

from skcriteria import data


# =============================================================================
# CONSTANTS
# =============================================================================


CHARS = tuple(string.ascii_letters + string.digits)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def data_values():
    def make(seed=None):
        random = np.random.default_rng(seed=seed)

        alts = random.integers(2, 10)
        crit = random.integers(2, 10)

        mtx = random.random((alts, crit))

        criteria = random.choice(list(data.CRITERIA_ALIASES), crit)
        weights = random.random(crit)

        anames = ["A." + "".join(random.choice(CHARS, 5)) for _ in range(alts)]
        cnames = ["C." + "".join(random.choice(CHARS, 5)) for _ in range(crit)]

        return mtx, criteria, weights, anames, cnames

    return make


# =============================================================================
# MUST WORK
# =============================================================================


def test_simple_creation(data_values):

    mtx, criteria, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    np.testing.assert_array_equal(dm.mtx, mtx)
    np.testing.assert_array_equal(dm.criteria, data.ascriteria(criteria))
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_weights(data_values):
    mtx, criteria, _, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        anames=anames,
        cnames=cnames,
    )

    weights = np.ones(len(dm.criteria))

    np.testing.assert_array_equal(dm.mtx, mtx)
    np.testing.assert_array_equal(dm.criteria, data.ascriteria(criteria))
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_anames(data_values):

    mtx, criteria, weights, _, cnames = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        cnames=cnames,
    )

    anames = [f"A{idx}" for idx in range(mtx.shape[0])]

    np.testing.assert_array_equal(dm.mtx, mtx)
    np.testing.assert_array_equal(dm.criteria, data.ascriteria(criteria))
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_cnames(data_values):
    mtx, criteria, weights, anames, _ = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
    )

    cnames = [f"C{idx}" for idx in range(mtx.shape[1])]

    np.testing.assert_array_equal(dm.mtx, mtx)
    np.testing.assert_array_equal(dm.criteria, data.ascriteria(criteria))
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_cnames_and_anames(data_values):
    mtx, criteria, weights, _, _ = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
    )

    anames = [f"A{idx}" for idx in range(mtx.shape[0])]
    cnames = [f"C{idx}" for idx in range(mtx.shape[1])]

    np.testing.assert_array_equal(dm.mtx, mtx)
    np.testing.assert_array_equal(dm.criteria, data.ascriteria(criteria))
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_copy(data_values):
    mtx, criteria, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )
    copy = dm.copy()

    assert dm is not copy
    assert dm == copy


def test_self_eq(data_values):
    mtx, criteria, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )
    copy = dm

    assert dm is copy
    assert dm == copy


def test_self_ne(data_values):
    mtx, criteria, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    omtx, ocriteria, oweights, oanames, ocnames = data_values(seed=43)

    other = data.mkdm(
        mtx=omtx,
        criteria=ocriteria,
        weights=oweights,
        anames=oanames,
        cnames=ocnames,
    )
    assert dm != other


@pytest.mark.xfail
def test_simple_repr(data_values):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    assert repr(data)


@pytest.mark.xfail
def test_simple_html(data_values):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx,
        criteria=criteria,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    assert data._repr_html_()


# =============================================================================
# MUST FAIL
# =============================================================================


def test_no_provide_mtx(data_values):
    _, criteria, weights, anames, cnames = data_values(seed=42)
    with pytest.raises(TypeError):
        data.mkdm(
            criteria=criteria, weights=weights, cnames=cnames, anames=anames
        )



def test_no_provide_criteria(data_values):
    mtx, _, weights, anames, cnames = data_values(seed=42)
    with pytest.raises(TypeError):
        data.mkdm(
            mtxt=mtx, weights=weights, cnames=cnames, anames=anames
        )


@pytest.mark.xfail
def test_invalid_criteria(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    criteria = [1, 2, 3, 4]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_weight_no_float(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    weights = ["hola"]
    with pytest.raises(ValueError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_missmatch_criteria(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    criteria = criteria[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_missmatch_weights(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    weights = weights[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_missmatch_anames(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    anames = anames[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_missmatch_cnames(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    cnames = cnames[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_mtx_ndim1(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    mtx = mtx.flatten()
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


@pytest.mark.xfail
def test_mtx_ndim3(data_values):
    mtx, criteria, weights, anames, cnames = init_data
    mtx = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx,
            criteria=criteria,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )
