#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tests for skcriteria.tiebreaker"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg import RankResult, SKCDecisionMakerABC
from skcriteria.agg.simple import WeightedProductModel, WeightedSumModel
from skcriteria.tiebreaker import TieBreaker


# =============================================================================
# TESTS
# =============================================================================


def test_TieBreaker_properties():
    primary = WeightedSumModel()
    secondary = WeightedProductModel()

    tb = TieBreaker(primary, secondary, force=False)

    assert tb.dmaker is primary
    assert tb.untier is secondary
    assert tb.force is False


def test_TieBreaker_repr():
    primary = WeightedSumModel()
    secondary = WeightedProductModel()

    tb = TieBreaker(primary, secondary, force=False)

    expected = (
        "<TieBreaker dmaker=<WeightedSumModel []>, "
        "untier=<WeightedProductModel []>, force=False>"
    )

    assert repr(tb) == expected


def test_TieBreaker_bad_dmaker():
    primary = "Despair"
    secondary = WeightedProductModel()

    with pytest.raises(TypeError) as ex:
        tb = TieBreaker(primary, secondary, force=False)
        assert "'dmaker' must implement 'evaluate()' method" in str(ex.value)



def test_TieBreaker_bad_untier():
    primary = WeightedSumModel()
    secondary = "FIFA"

    with pytest.raises(TypeError) as ex:
        tb = TieBreaker(primary, secondary, force=False)
        assert "'untier' must implement 'evaluate()' method" in str(ex.value)


def test_TieBreaker():
    class Tier(SKCDecisionMakerABC):
        """Decision maker que devuelve [1,1,2,2,3] hardcodeado."""

        _skcriteria_parameters = []

        def _evaluate_data(self, **kwargs):
            return [1, 1, 2, 2, 3], {}

        def _make_result(self, alternatives, values, extra):
            return RankResult(
                method="ExamplePrimary",
                alternatives=alternatives,
                values=values,
                extra=extra,
            )

    # Crear la matriz de decisión
    dm = skc.mkdm(
        matrix=[
            [100, 8.5, 7.2],  # A
            [120, 8.5, 6.8],  # B - similar quality a A
            [150, 9.2, 8.1],  # C
            [140, 9.2, 7.9],  # D - similar quality a C
            [200, 9.8, 9.0],  # E - mejor en todo pero más caro
        ],
        objectives=[max, max, max],
    )

    dmaker = Tier()
    tb = TieBreaker(dmaker, WeightedSumModel(), force=False)

    orank = dmaker.evaluate(dm)
    rank = tb.evaluate(dm)

    np.testing.assert_array_equal(
        rank.extra_.tiebreaker.original_values, orank.values
    )

    np.testing.assert_array_equal(rank.alternatives, orank.alternatives)
    np.testing.assert_array_equal(rank.values, [2, 1, 3, 4, 5])
    assert not rank.extra_.tiebreaker.forced


def test_TieBreaker_no_ties():
    class Tier(SKCDecisionMakerABC):
        """Decision maker que devuelve [1,1,2,2,3] hardcodeado."""

        _skcriteria_parameters = []

        def _evaluate_data(self, **kwargs):
            return [5, 4, 3, 2, 1], {}

        def _make_result(self, alternatives, values, extra):
            return RankResult(
                method="ExamplePrimary",
                alternatives=alternatives,
                values=values,
                extra=extra,
            )

    # Crear la matriz de decisión
    dm = skc.mkdm(
        matrix=[
            [100, 5, 5],  # A
            [100, 10, 10],  # B
            [100, 15, 15],  # C
            [100, 20, 20],  # D
            [100, 25, 25],  # E
        ],
        objectives=[max, max, max],
    )

    dmaker = Tier()
    tb = TieBreaker(dmaker, WeightedSumModel(), force=False)

    orank = dmaker.evaluate(dm)
    rank = tb.evaluate(dm)

    assert orank == rank
    np.testing.assert_array_equal(rank.alternatives, orank.alternatives)
    np.testing.assert_array_equal(rank.values, [5,4,3,2,1])


def test_TieBreaker_forced():
    class Tier(SKCDecisionMakerABC):
        """Decision maker que devuelve [1,1,2,2,3] hardcodeado."""

        _skcriteria_parameters = []

        def _evaluate_data(self, alternatives, **kwargs):
            return np.ones_like(alternatives), {}

        def _make_result(self, alternatives, values, extra):
            return RankResult(
                method="Tier",
                alternatives=alternatives,
                values=values,
                extra=extra,
            )

    # Crear la matriz de decisión
    dm = skc.mkdm(
        matrix=[
            [100, 8.5, 7.2],  # A
            [120, 8.5, 6.8],  # B - similar quality a A
            [150, 9.2, 8.1],  # C
            [140, 9.2, 7.9],  # D - similar quality a C
            [200, 9.8, 9.0],  # E - mejor en todo pero más caro
        ],
        objectives=[max, max, max],
    )

    dmaker = Tier()
    tb = TieBreaker(dmaker, dmaker, force=True)

    orank = dmaker.evaluate(dm)
    rank = tb.evaluate(dm)

    np.testing.assert_array_equal(
        rank.extra_.tiebreaker.original_values, orank.values
    )

    np.testing.assert_array_equal(rank.alternatives, orank.alternatives)
    np.testing.assert_array_equal(rank.values, [1, 2, 3, 4, 5])
    assert rank.extra_.tiebreaker.forced
