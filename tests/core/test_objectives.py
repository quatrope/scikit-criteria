#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.core.objectives"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria.core import objectives

# =============================================================================
# ENUM
# =============================================================================


def test_Objective_from_alias():
    for alias in objectives.Objective._MAX_ALIASES.value:
        objective = objectives.Objective.from_alias(alias)
        assert objective is objectives.Objective.MAX
    for alias in objectives.Objective._MIN_ALIASES.value:
        objective = objectives.Objective.from_alias(alias)
        assert objective is objectives.Objective.MIN
    with pytest.raises(ValueError):
        objectives.Objective.from_alias("no anda")


def test_Objective_eq():
    # Assert MIN is equal to all MIN aliases and not equal to all MAX aliases
    for alias in objectives.Objective._MIN_ALIASES.value:
        assert objectives.Objective.MIN == alias
        assert alias == objectives.Objective.MIN
        assert objectives.Objective.MAX != alias
        assert alias != objectives.Objective.MAX

    # Assert MAX is equal to all MAX aliases and not equal to all MIN aliases
    for alias in objectives.Objective._MAX_ALIASES.value:
        assert objectives.Objective.MAX == alias
        assert alias == objectives.Objective.MAX
        assert objectives.Objective.MIN != alias
        assert alias != objectives.Objective.MIN

    assert objectives.Objective.MIN != "whatever"
    assert objectives.Objective.MAX != "whatever"


def test_Objective_str():
    assert str(objectives.Objective.MAX) == objectives.Objective.MAX.name
    assert str(objectives.Objective.MIN) == objectives.Objective.MIN.name


def test_Objective_to_symbol():
    assert (
        objectives.Objective.MAX.to_symbol()
        == objectives.Objective._MAX_STR.value
    )
    assert (
        objectives.Objective.MIN.to_symbol()
        == objectives.Objective._MIN_STR.value
    )


# =============================================================================
# DEPRECATED
# =============================================================================


def test_Objective_to_string():
    with pytest.deprecated_call():
        assert (
            objectives.Objective.MAX.to_string()
            == objectives.Objective.MAX.to_symbol()
        )
    with pytest.deprecated_call():
        assert (
            objectives.Objective.MIN.to_string()
            == objectives.Objective.MIN.to_symbol()
        )


def test_Objective_construct_from_alias():
    with pytest.deprecated_call():
        objectives.Objective.construct_from_alias(max)
        objectives.Objective.construct_from_alias(min)
