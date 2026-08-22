#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.importance._base_importance"""

# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria import mkdm
from skcriteria.agg import simple
from skcriteria.importance._base_importance import CriteriaImportanceABC

# =============================================================================
# HELPERS
# =============================================================================


def _dm():
    return mkdm(
        matrix=[[1, 3], [2, 2], [3, 1]],
        objectives=[max, max],
        alternatives=["A", "B", "C"],
        criteria=["C0", "C1"],
    )


# =============================================================================
# ABSTRACTNESS
# =============================================================================


def test_CriteriaImportanceABC_not_instantiable():
    with pytest.raises(TypeError):
        CriteriaImportanceABC(simple.WeightedSumModel())


def test_CriteriaImportanceABC_subclass_not_redefining_evaluate_subproblem():
    class Foo(CriteriaImportanceABC):
        pass

    with pytest.raises(TypeError):
        Foo(simple.WeightedSumModel())


def test_CriteriaImportanceABC_evaluate_subproblem_not_implemented():
    class Foo(CriteriaImportanceABC):
        _invert_similarity = True

        def _evaluate_subproblem(self, dm, criterion):
            return super()._evaluate_subproblem(dm, criterion)

    checker = Foo(simple.WeightedSumModel())
    with pytest.raises(NotImplementedError):
        checker.evaluate(_dm())


def test_CriteriaImportanceABC_subclass_missing_invert_similarity():
    class Foo(CriteriaImportanceABC):
        def _evaluate_subproblem(self, dm, criterion):
            keep = [c for c in dm.criteria if c != criterion]
            rank_sub = self._dmaker.evaluate(dm[keep])
            return f"FOO-{criterion}", rank_sub, {}

    checker = Foo(simple.WeightedSumModel())
    with pytest.raises(AttributeError):
        checker.evaluate(_dm())


# =============================================================================
# PROPERTIES
# =============================================================================


def test_CriteriaImportanceABC_properties():
    dmaker = simple.WeightedSumModel()

    class Foo(CriteriaImportanceABC):
        _invert_similarity = True

        def _evaluate_subproblem(self, dm, criterion):
            keep = [c for c in dm.criteria if c != criterion]
            rank_sub = self._dmaker.evaluate(dm[keep])
            return f"FOO-{criterion}", rank_sub, {}

    checker = Foo(
        dmaker,
        metric="kendall",
        untied=True,
        allow_missing_alternatives=False,
        preferred_parallel_backend="threads",
        n_jobs=2,
    )

    assert checker.dmaker is dmaker
    assert checker.metric == "kendall"
    assert checker.untied is True
    assert checker.allow_missing_alternatives is False
    assert checker.preferred_parallel_backend == "threads"
    assert checker.n_jobs == 2
