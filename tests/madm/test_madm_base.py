#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.madm._base."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy
from collections import Counter

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


class test_ResultBase_skacriteria_result_series_no_defined:
    with pytest.raises(TypeError):

        class Foo(ResultABC):
            def _validate_result(self, values):
                pass


class test_ResultBase_original_validare_result_fail:
    class Foo(ResultABC):
        _skcriteria_result_series = "foo"

        def _validate_result(self, values):
            return super()._validate_result(values)

    with pytest.raises(NotImplementedError):
        Foo("foo", ["abc"], [1, 2, 3], {})


def test_ResultBase_repr():
    class TestResult(ResultABC):
        _skcriteria_result_series = "foo"

        def _validate_result(self, values):
            pass

    method = "test_method"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = TestResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    expected = (
        "Alternatives  a  b  c\nfoo           1  2  3\n[Method: test_method]"
    )

    assert repr(result) == expected


def test_ResultBase_repr_html():
    class TestResult(ResultABC):
        _skcriteria_result_series = "foo"

        def _validate_result(self, values):
            pass

    method = "test_method"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = PyQuery(
        TestResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )._repr_html_()
    )

    expected = PyQuery(
        """
        <div class='skcresult skcresult-foo'>
        <table id="T_cc7f5_" >
            <thead>
            <tr>
                <th class="blank level0" >Alternatives</th>
                <th class="col_heading level0 col0" >a</th>
                <th class="col_heading level0 col1" >b</th>
                <th class="col_heading level0 col2" >c</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <th id="T_cc7f5_level0_row0" class="row_heading level0 row0" >
                    foo
                </th>
                <td id="T_cc7f5_row0_col0" class="data row0 col0" >1</td>
                <td id="T_cc7f5_row0_col1" class="data row0 col1" >2</td>
                <td id="T_cc7f5_row0_col2" class="data row0 col2" >3</td>
            </tr>
            </tbody>
        </table>
        <em class='skcresult-method'>Method: test_method</em>
        </div>
        """
    )
    result_html = result.remove("style").text()
    expected_html = expected.remove("style").text()

    assert result_html == expected_html


def test_ResultBase_equals():
    class TestResult(ResultABC):
        _skcriteria_result_series = "foo"

        def _validate_result(self, values):
            pass

    method = "test_method"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1, "beta": np.array([1, 2])}

    result = TestResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    # the same
    assert result.values_equals(result)
    assert result == result
    assert result.equals(result)
    assert result.aequals(result)

    # copy
    rcopy = copy.deepcopy(result)

    assert result.values_equals(rcopy)
    assert result == rcopy
    assert result.equals(rcopy)
    assert result.aequals(rcopy)

    # not equals
    neq = TestResult(
        method=method, alternatives=alternatives, values=rank, extra={}
    )

    assert result.values_equals(neq)
    assert result != neq
    assert not result.equals(neq)
    assert not result.aequals(neq)


# =============================================================================
# RANK RESULT
# =============================================================================


@pytest.mark.parametrize(
    "rank, has_ties, untied_rank",
    [
        ([1, 2, 3], False, [1, 2, 3]),
        ([1, 2, 1], True, [1, 3, 2]),
    ],
)
def test_RankResult(rank, has_ties, untied_rank):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    result = RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )
    assert result.has_ties_ == has_ties
    assert result.ties_ == Counter(result.rank_)
    assert np.all(result.method == method)
    assert np.all(result.alternatives == alternatives)
    assert np.all(result.rank_ == rank)
    assert np.all(result.extra_ == result.e_ == extra)
    assert np.all(result.untied_rank_ == untied_rank)

    result_as_series = result.to_series()
    assert np.all(result_as_series.index == alternatives)
    assert np.all(result_as_series.to_numpy() == rank)
    assert np.all(result_as_series.name == "Rank")

    result_as_series_untied = result.to_series(untied=True)
    assert np.all(result_as_series_untied.index == alternatives)
    assert np.all(result_as_series_untied.to_numpy() == untied_rank)
    assert np.all(result_as_series_untied.name == "Untied rank")


@pytest.mark.parametrize("rank", [[1, 2, 5], [1, 2]])
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

    assert result.shape == (length,)


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


# =============================================================================
# KERNEL
# =============================================================================


@pytest.mark.parametrize(
    "kernel, kernel_size, kernel_where, kernel_alternatives",
    [
        ([False, False, False], 0, [], []),
        ([False, False, True], 1, [2], ["c"]),
        ([True, True, False], 2, [0, 1], ["a", "b"]),
        ([True, True, True], 3, [0, 1, 2], ["a", "b", "c"]),
    ],
)
def test_KernelResult(kernel, kernel_size, kernel_where, kernel_alternatives):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    result = KernelResult(
        method=method, alternatives=alternatives, values=kernel, extra=extra
    )

    assert np.all(result.method == method)
    assert np.all(result.alternatives == alternatives)
    assert np.all(result.extra_ == result.e_ == extra)
    assert np.all(result.kernel_ == kernel)
    assert np.all(result.kernel_size_ == kernel_size)
    assert np.all(result.kernel_where_ == kernel_where)
    assert np.all(result.kernel_alternatives_ == kernel_alternatives)

    result_as_series = result.to_series()
    assert np.all(result_as_series.index == alternatives)
    assert np.all(result_as_series.to_numpy() == kernel)
    assert np.all(result_as_series.name == "Kernel")

    with pytest.deprecated_call():
        assert np.all(result.kernelwhere_ == kernel_where)


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
