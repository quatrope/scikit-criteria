#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg._base."""


# =============================================================================
# IMPORTS
# =============================================================================

import copy
from collections import Counter

import numpy as np

from pyquery import PyQuery

import pytest

from skcriteria.agg import (
    SKCRankResult,
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

    ranker = Foo()

    result = ranker.evaluate(dm)

    assert np.all(result.alternatives == dm.alternatives)
    assert np.all(result.values == np.arange(len(dm.alternatives)) + 1)
    assert result.extra_ == {}


@pytest.mark.parametrize("not_redefine", ["_evaluate_data"])
def test_SKCDecisionMakerABC_not_redefined(not_redefine):
    content = {"_skcriteria_parameters": []}
    for method_name in ["_evaluate_data"]:
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

    ranker = Foo()

    with pytest.raises(NotImplementedError):
        ranker.evaluate(dm)


# =============================================================================
# SKCRankResult
# =============================================================================


def test_SKCRankResult_repr():
    method = "test_method"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = SKCRankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    expected = (
        "Alternatives  a  b  c\nRank          1  2  3\n[Method: test_method]"
    )

    assert repr(result) == expected


def test_SKCRankResult_repr_html():
    method = "test_method"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = PyQuery(
        SKCRankResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )._repr_html_()
    )

    expected = PyQuery(
        """
        <div class='skcresult skcresult-rank'>
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
                    Rank
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


def test_SKCRankResult_equals():
    method = "test_method"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1, "beta": np.array([1.0, 2.0])}

    result = SKCRankResult(
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
    neq = SKCRankResult(
        method=method, alternatives=alternatives, values=rank, extra={}
    )

    assert result.values_equals(neq)
    assert result != neq
    assert not result.equals(neq)
    assert not result.aequals(neq)

    # tolerance
    slightly_off = SKCRankResult(
        method=method,
        alternatives=alternatives,
        values=rank,
        extra={
            "alfa": extra["alfa"],
            "beta": np.array([1.01, 2.01]),
        },
    )

    assert not result.aequals(slightly_off)
    assert result.aequals(slightly_off, rtol=1e-01, atol=1e-01)


# =============================================================================
# RANK RESULT (using SKCRankResult)
# =============================================================================


@pytest.mark.parametrize(
    "rank, has_ties, untied_rank",
    [
        ([1, 2, 3], False, [1, 2, 3]),
        ([1, 2, 1], True, [1, 3, 2]),
    ],
)
def test_SKCRankResult_ranking_functionality(rank, has_ties, untied_rank):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    result = SKCRankResult(
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
    assert np.all(result_as_series.name == "Result")

    result_as_series_untied = result.to_series(untied=True)
    assert np.all(result_as_series_untied.index == alternatives)
    assert np.all(result_as_series_untied.to_numpy() == untied_rank)
    assert np.all(result_as_series_untied.name == "Untied Rank")


@pytest.mark.parametrize("rank", [[1, 2, 5], [1, 2]])
def test_SKCRankResult_invalid_rank(rank):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    with pytest.raises(ValueError):
        SKCRankResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )


def test_SKCRankResult_shape():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    alternatives = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = SKCRankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert result.shape == (length,)


def test_SKCRankResult_len():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    alternatives = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = SKCRankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert len(result) == length


# =============================================================================
# KERNEL FUNCTIONALITY (using SKCRankResult)
# =============================================================================


@pytest.mark.parametrize(
    "kernel_mask, kernel_size, kernel_where, kernel_alternatives",
    [
        ([False, False, False], 0, [], []),
        ([False, False, True], 1, [2], ["c"]),
        ([True, True, False], 2, [0, 1], ["a", "b"]),
        ([True, True, True], 3, [0, 1, 2], ["a", "b", "c"]),
    ],
)
def test_SKCRankResult_kernel_functionality(
    kernel_mask, kernel_size, kernel_where, kernel_alternatives
):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    # Create result using from_kernel factory method
    result = SKCRankResult.from_kernel(
        method=method,
        alternatives=alternatives,
        values=kernel_mask,
        extra=extra,
    )

    assert np.all(result.method == method)
    assert np.all(result.alternatives == alternatives)
    assert np.all(result.extra_ == result.e_ == extra)

    # Test kernel methods
    assert np.all(result.kernel() == kernel_mask)
    assert result.kernel_size() == kernel_size
    assert np.all(result.kernel_where() == kernel_where)
    assert np.all(result.kernel_alternatives() == kernel_alternatives)

    # Test deprecated properties with warnings
    with pytest.deprecated_call():
        assert np.all(result.kernel_ == kernel_mask)

    with pytest.deprecated_call():
        assert result.kernel_size_ == kernel_size

    with pytest.deprecated_call():
        assert np.all(result.kernel_where_ == kernel_where)

    with pytest.deprecated_call():
        assert np.all(result.kernel_alternatives_ == kernel_alternatives)

    with pytest.deprecated_call():
        assert np.all(result.kernelwhere_ == kernel_where)


def test_SKCRankResult_kernel_representation():
    method = "foo"
    alternatives = ["a", "b", "c"]
    kernel_mask = [True, False, True]
    extra = {"alfa": 1}

    result = SKCRankResult.from_kernel(
        method=method,
        alternatives=alternatives,
        values=kernel_mask,
        extra=extra,
    )

    # Test kernel string representation
    kernel_string = result.to_kernel_string()
    expected_kernel_string = (
        "Alternatives     a      b     c\n"
        "Kernel        True  False  True\n"
        "[Method: foo]"
    )

    assert kernel_string == expected_kernel_string

    # Test kernel HTML representation
    kernel_html = result.to_kernel_html()
    assert "skcresult-kernel" in kernel_html
    assert "Method: foo" in kernel_html


def test_SKCRankResult_multi_level_kernel():
    """Test kernel functionality with multi-level rankings."""
    method = "foo"
    alternatives = ["a", "b", "c", "d"]
    rank = [1, 2, 3, 1]  # Two alternatives at rank 1
    extra = {}

    result = SKCRankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    # Test with different kernel thresholds
    kernel_1 = result.kernel(max_in_kernel=1)
    expected_kernel_1 = [True, False, False, True]  # ranks 1
    assert np.all(kernel_1 == expected_kernel_1)
    assert result.kernel_size(max_in_kernel=1) == 2

    kernel_2 = result.kernel(max_in_kernel=2)
    expected_kernel_2 = [True, True, False, True]  # ranks 1 and 2
    assert np.all(kernel_2 == expected_kernel_2)
    assert result.kernel_size(max_in_kernel=2) == 3


@pytest.mark.parametrize("values", [[1, 2, 5], ["a", "b", "c"]])
def test_SKCRankResult_invalid_values(values):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    with pytest.raises(ValueError):
        SKCRankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


def test_SKCRankResult_from_kernel_validation():
    """Test that from_kernel properly converts boolean masks."""
    method = "foo"
    alternatives = ["a", "b", "c"]
    kernel_mask = [True, False, True]
    extra = {}

    result = SKCRankResult.from_kernel(
        method=method,
        alternatives=alternatives,
        values=kernel_mask,
        extra=extra,
    )

    # Should convert True->1, False->2
    expected_values = [1, 2, 1]
    assert np.all(result.values == expected_values)

    # Original kernel should be recoverable
    assert np.all(result.kernel() == kernel_mask)
