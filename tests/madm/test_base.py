#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.madm._base."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from pyquery import PyQuery

import pytest

from skcriteria.madm import (
    KernelResult,
    RankResult,
    ResultABC,
    SKCDecisionMakerABC,
)

# =============================================================================
# SKCDecisionMakerABC
# =============================================================================


def test_SKCDecisionMakerABC_flow(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(SKCDecisionMakerABC):
        _skcriteria_parameters = []

        def _evaluate_data(self, alternatives, **kwargs):
            return np.arange(len(alternatives)) + 1, {}

        def _make_result(self, alternatives, values, extra):
            return {
                "alternatives": alternatives,
                "rank": values,
                "extra": extra,
            }

    ranker = Foo()

    result = ranker.evaluate(dm)

    assert np.all(result["alternatives"] == dm.alternatives)
    assert np.all(result["rank"] == np.arange(len(dm.alternatives)) + 1)
    assert result["extra"] == {}


@pytest.mark.parametrize("not_redefine", ["_evaluate_data", "_make_result"])
def test_SKCDecisionMakerABC_not_redefined(not_redefine):
    content = {"_skcriteria_parameters": []}
    for method_name in ["_evaluate_data", "_make_result", "_validate_data"]:
        if method_name != not_redefine:
            content[method_name] = lambda **kws: None

    Foo = type("Foo", (SKCDecisionMakerABC,), content)

    with pytest.raises(TypeError):
        Foo()


def test_SKCDecisionMakerABC_evaluate_data_not_implemented(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(SKCDecisionMakerABC):
        _skcriteria_parameters = []

        def _evaluate_data(self, **kwargs):
            super()._evaluate_data(**kwargs)

        def _make_result(self, alternatives, values, extra):
            return {
                "alternatives": alternatives,
                "rank": values,
                "extra": extra,
            }

    ranker = Foo()

    with pytest.raises(NotImplementedError):
        ranker.evaluate(dm)


def test_SKCDecisionMakerABC_make_result_not_implemented(decision_matrix):

    dm = decision_matrix(seed=42)

    class Foo(SKCDecisionMakerABC):
        _skcriteria_parameters = []

        def _evaluate_data(self, alternatives, **kwargs):
            return np.arange(len(alternatives)) + 1, {}

        def _make_result(self, **kwargs):
            super()._make_result(**kwargs)

    ranker = Foo()

    with pytest.raises(NotImplementedError):
        ranker.evaluate(dm)


# =============================================================================
# RESULT BASE
# =============================================================================


class test_ResultBase_skacriteria_result_column_no_defined:

    with pytest.raises(TypeError):

        class Foo(ResultABC):
            def _validate_result(self, values):
                pass


class test_ResultBase_original_validare_result_fail:
    class Foo(ResultABC):
        _skcriteria_result_column = "foo"

        def _validate_result(self, values, allow_ties):
            return super()._validate_result(values, allow_ties)

    with pytest.raises(NotImplementedError):
        Foo("foo", ["abc"], [1, 2, 3], {})


# =============================================================================
# RANK RESULT
# =============================================================================


def test_RankResult():
    method = "foo"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert np.all(result.method == method)
    assert np.all(result.alternatives == alternatives)
    assert np.all(result.rank_ == rank)
    assert np.all(result.extra_ == result.e_ == extra)
    assert np.all(result.untied_rank_ == rank)


def test_RankResult_ties():
    method = "foo"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 1]
    extra = {"alfa": 1}

    result = RankResult(
        method=method,
        alternatives=alternatives,
        values=rank,
        extra=extra,
        allow_ties=True,
    )

    assert np.all(result.method == method)
    assert np.all(result.alternatives == alternatives)
    assert np.all(result.rank_ == rank)
    assert np.all(result.extra_ == result.e_ == extra)
    assert np.all(result.untied_rank_ == [1, 3, 2])


@pytest.mark.parametrize("rank", [[1, 2, 5], [1, 1, 1], [1, 2, 2], [1, 2]])
def test_RankResult_invalid_rank(rank):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    with pytest.raises(ValueError):
        RankResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )


def test_RankResult_shape():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    alternatives = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert result.shape == (length, 1)


def test_RankResult_len():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    alternatives = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert len(result) == length


def test_RankResult_repr():
    method = "foo"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    expected = "      a  b  c\n" "Rank  1  2  3\n" "[Method: foo]"

    assert repr(result) == expected


def test_RankResult_repr_html():
    method = "foo"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = PyQuery(
        RankResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )._repr_html_()
    )

    expected = PyQuery(
        """
        <div class='skcresult skcresult-rank'>
        <table id="T_cc7f5_" >
            <thead>
            <tr>
                <th class="blank level0" ></th>
                <th class="col_heading level0 col0" >a</th>
                <th class="col_heading level0 col1" >b</th>
                <th class="col_heading level0 col2" >c</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <th id="T_cc7f5_level0_row0" class="row_heading level0 row0" >
                    Rank
                </th>
                <td id="T_cc7f5_row0_col0" class="data row0 col0" >1</td>
                <td id="T_cc7f5_row0_col1" class="data row0 col1" >2</td>
                <td id="T_cc7f5_row0_col2" class="data row0 col2" >3</td>
            </tr>
            </tbody>
        </table>
        <em class='rankresult-method'>Method: foo</em>
        </div>
        """
    )
    assert result.remove("style").text() == expected.remove("style").text()


# =============================================================================
# KERNEL
# =============================================================================


@pytest.mark.parametrize("values", [[1, 2, 5], [True, False, 1], [1, 2, 3]])
def test_KernelResult_invalid_rank(values):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    with pytest.raises(ValueError):
        KernelResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


def test_KernelResult_repr_html():
    method = "foo"
    alternatives = ["a", "b", "c"]
    rank = [True, False, True]
    extra = {"alfa": 1}

    result = PyQuery(
        KernelResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )._repr_html_()
    )

    expected = PyQuery(
        """
        <div class='rankresult'>
        <table id="T_cc7f5_" >
            <thead>
            <tr>
                <th class="blank level0" ></th>
                <th class="col_heading level0 col0" >a</th>
                <th class="col_heading level0 col1" >b</th>
                <th class="col_heading level0 col2" >c</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <th id="T_cc7f5_level0_row0" class="row_heading level0 row0" >
                    Kernel
                </th>
                <td id="T_cc7f5_row0_col0" class="data row0 col0" >True</td>
                <td id="T_cc7f5_row0_col1" class="data row0 col1" >False</td>
                <td id="T_cc7f5_row0_col2" class="data row0 col2" >True</td>
            </tr>
            </tbody>
        </table>
        <em class='rankresult-method'>Method: foo</em>
        </div>
        """
    )

    assert result.remove("style").text() == expected.remove("style").text()
