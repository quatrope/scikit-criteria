#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
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

from skcriteria import Data, DataValidationError, ascriteria


# =============================================================================
# CONSTANTS
# =============================================================================

CHARS = tuple(string.ascii_letters + string.digits)

RANDOM = np.random.RandomState(42)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def init_data():
    random = RANDOM

    alts = random.randint(2, 10)
    crit = random.randint(2, 10)

    mtx = random.rand(alts, crit)

    criteria = random.choice([min, max, 1, -1, np.min, np.max], crit)
    weights = random.rand(crit)

    anames = ["".join(random.choice(CHARS, 5)) for _ in range(alts)]
    cnames = ["".join(random.choice(CHARS, 5)) for _ in range(crit)]

    return mtx, criteria, weights, anames, cnames


# =============================================================================
# MUST WORK
# =============================================================================

def test_simple_creation(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)

    assert np.all(data.mtx == mtx)
    assert np.all(data.criteria == ascriteria(criteria))
    assert np.all(data.weights == weights)
    assert data.anames == tuple(anames)
    assert data.cnames == tuple(cnames)


def test_no_provide_weights(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, anames=anames, cnames=cnames)

    assert np.all(data.mtx == mtx)
    assert np.all(data.criteria == ascriteria(criteria))
    assert data.weights is None
    assert data.anames == tuple(anames)
    assert data.cnames == tuple(cnames)


def test_no_provide_anames(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    anames = (f'A{idx}' for idx in range(mtx.shape[0]))

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights, cnames=cnames)

    assert np.all(data.mtx == mtx)
    assert np.all(data.criteria == ascriteria(criteria))
    assert np.all(data.weights == weights)
    assert data.anames == tuple(anames)
    assert data.cnames == tuple(cnames)


def test_no_provide_cnames(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    cnames = (f'C{idx}' for idx in range(mtx.shape[1]))

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights, anames=anames)

    assert np.all(data.mtx == mtx)
    assert np.all(data.criteria == ascriteria(criteria))
    assert np.all(data.weights == weights)
    assert data.anames == tuple(anames)
    assert data.cnames == tuple(cnames)


def test_no_provide_cnames_and_anames(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    anames = (f'A{idx}' for idx in range(mtx.shape[0]))
    cnames = (f'C{idx}' for idx in range(mtx.shape[1]))

    data = Data(mtx=mtx, criteria=criteria, weights=weights)

    assert np.all(data.mtx == mtx)
    assert np.all(data.criteria == ascriteria(criteria))
    assert np.all(data.weights == weights)
    assert data.anames == tuple(anames)
    assert data.cnames == tuple(cnames)


def test_copy(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)
    copy = data.copy()

    assert data is not copy
    assert data == copy


def test_self_eq(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)
    copy = data

    assert data is copy
    assert data == copy


def test_self_ne(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)
    other = Data(
        mtx=mtx, criteria=criteria, weights=weights)
    assert data != other


def test_simple_repr(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)

    assert repr(data)


def test_simple_html(init_data):
    mtx, criteria, weights, anames, cnames = init_data

    data = Data(
        mtx=mtx, criteria=criteria, weights=weights,
        anames=anames, cnames=cnames)

    assert data._repr_html_()


# =============================================================================
# MUST FAIL
# =============================================================================

def test_no_provide_mtx(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    with pytest.raises(TypeError):
        Data(criteria=criteria, weights=weights, cnames=cnames, anames=anames)


def test_no_provide_criteria(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    with pytest.raises(TypeError):
        Data(mtx=mtx, weights=weights, anames=anames, cnames=cnames)


def test_invalid_criteria(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    criteria = [1, 2, 3, 4]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_weight_no_float(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    weights = ["hola"]
    with pytest.raises(ValueError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_missmatch_criteria(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    criteria = criteria[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_missmatch_weights(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    weights = weights[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_missmatch_anames(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    anames = anames[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_missmatch_cnames(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    cnames = cnames[1:]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_mtx_ndim1(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    mtx = mtx.flatten()
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)


def test_mtx_ndim3(init_data):
    mtx, criteria, weights, anames, cnames = init_data
    mtx = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    with pytest.raises(DataValidationError):
        Data(
            mtx=mtx, criteria=criteria, weights=weights,
            anames=anames, cnames=cnames)
