#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.bunch"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria.utils import rank


# =============================================================================
# TEST Bunch
# =============================================================================


def test_rank():
    values = [0.5, 0.2, 0.6, 0.8]
    expected = [2, 1, 3, 4]
    result = rank.rank_values(values)
    assert np.all(result == expected)


def test_rank_reverse():
    values = [0.5, 0.2, 0.6, 0.8]
    expected = [3, 4, 2, 1]
    result = rank.rank_values(values, reverse=True)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "ra, rb",
    [
        ([11, 20, 14], [11, 20, 14]),
        ([11, 20, 14], [14, 16, 15]),
        ([11, 20, 14], [15, 19, 12]),
        ([14, 16, 15], [15, 19, 12]),
    ],
)
def test_dominance(ra, rb):
    result = rank.dominance(ra, rb, reverse=False)
    assert result.eq == np.equal(ra, rb).sum()
    assert result.aDb == np.greater(ra, rb).sum()
    assert result.bDa == np.greater(rb, ra).sum()

    assert np.all(result.eq_where == np.equal(ra, rb))
    assert np.all(result.aDb_where == np.greater(ra, rb))
    assert np.all(result.bDa_where == np.greater(rb, ra))


@pytest.mark.parametrize(
    "ra, rb",
    [
        ([11, 20, 14], [11, 20, 14]),
        ([11, 20, 14], [14, 16, 15]),
        ([11, 20, 14], [15, 19, 12]),
        ([14, 16, 15], [15, 19, 12]),
    ],
)
def test_dominance_reverse(ra, rb):
    result = rank.dominance(ra, rb, reverse=True)

    assert result.eq == np.equal(ra, rb).sum()
    assert result.aDb == np.less(ra, rb).sum()
    assert result.bDa == np.less(rb, ra).sum()

    assert np.all(result.eq_where == np.equal(ra, rb))
    assert np.all(result.aDb_where == np.less(ra, rb))
    assert np.all(result.bDa_where == np.less(rb, ra))


def test_dominance_fail():
    with pytest.raises(ValueError):
        rank.dominance([1], [1, 2])
    with pytest.raises(ValueError):
        rank.dominance([3, 4], [1, 2], [True])
