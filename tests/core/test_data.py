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

import numpy as np

import pandas as pd

from pyquery import PyQuery

import pytest

from skcriteria.core import data


# =============================================================================
# HELPER
# =============================================================================


def construct_objectives_values(arr):
    return [data.Objective.construct_from_alias(obj).value for obj in arr]


def construct_objectives(arr):
    return [data.Objective.construct_from_alias(obj) for obj in arr]


# =============================================================================
# ENUM
# =============================================================================


def test_objective_construct():
    for alias in data.Objective._MAX_ALIASES.value:
        objective = data.Objective.construct_from_alias(alias)
        assert objective is data.Objective.MAX
    for alias in data.Objective._MIN_ALIASES.value:
        objective = data.Objective.construct_from_alias(alias)
        assert objective is data.Objective.MIN
    with pytest.raises(ValueError):
        data.Objective.construct_from_alias("no anda")


def test_objective_str():
    assert str(data.Objective.MAX) == data.Objective.MAX.name
    assert str(data.Objective.MIN) == data.Objective.MIN.name


def test_objective_to_string():
    assert data.Objective.MAX.to_string() == data.Objective._MAX_STR.value
    assert data.Objective.MIN.to_string() == data.Objective._MIN_STR.value


# =============================================================================
# MUST WORK
# =============================================================================


def test_DecisionMatrix_simple_creation(data_values):

    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)
    np.testing.assert_array_equal(dm.dtypes, [np.float64] * len(criteria))


def test_DecisionMatrix_no_provide_weights(data_values):
    mtx, objectives, _, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        alternatives=alternatives,
        criteria=criteria,
    )

    weights = np.ones(len(dm.objectives))

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)


def test_DecisionMatrix_no_provide_anames(data_values):

    mtx, objectives, weights, _, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        criteria=criteria,
    )

    alternatives = [f"A{idx}" for idx in range(mtx.shape[0])]

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)


def test_DecisionMatrix_no_provide_cnames(data_values):
    mtx, objectives, weights, alternatives, _ = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
    )

    criteria = [f"C{idx}" for idx in range(mtx.shape[1])]

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)


def test_DecisionMatrix_no_provide_cnames_and_anames(data_values):
    mtx, objectives, weights, _, _ = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
    )

    alternatives = [f"A{idx}" for idx in range(mtx.shape[0])]
    criteria = [f"C{idx}" for idx in range(mtx.shape[1])]

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)


# =============================================================================
# DECISION MATRIX
# =============================================================================


def test_DecisionMatrix_copy(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )
    copy = dm.copy()

    assert dm is not copy
    assert dm.equals(copy)


def test_DecisionMatrix_to_dataframe(data_values):

    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    df = dm.to_dataframe()

    rows = np.vstack((construct_objectives(objectives), weights, mtx))
    expected = pd.DataFrame(
        rows, index=["objectives", "weights"] + alternatives, columns=criteria
    )

    pd.testing.assert_frame_equal(df, expected)


def test_DecisionMatrix_to_dict(data_values):

    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    expected = {
        "matrix": mtx,
        "objectives": construct_objectives_values(objectives),
        "weights": weights,
        "alternatives": np.asarray(alternatives),
        "criteria": np.asarray(criteria),
        "dtypes": np.full(len(weights), float),
    }

    result = dm.to_dict()

    cmp = {k: (np.all(result[k] == expected[k])) for k in result.keys()}
    assert np.all(cmp.values())


def test_DecisionMatrix_describe(data_values):

    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    expected = pd.DataFrame(
        mtx, columns=criteria, index=alternatives
    ).describe()

    result = dm.describe()

    assert result.equals(expected)


# =============================================================================
# CMP
# =============================================================================


def test_DecisionMatrix_len(decision_matrix):
    dm1 = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    dm2 = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=30,
        max_criteria=30,
    )

    dm3 = decision_matrix(
        seed=42,
        min_alternatives=100,
        max_alternatives=100,
        min_criteria=30,
        max_criteria=30,
    )

    assert len(dm1) == len(dm2)
    assert len(dm1) != len(dm3) and len(dm2) != len(dm3)


def test_DecisionMatrix_shape(decision_matrix):
    dm1 = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    dm2 = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    dm3 = decision_matrix(
        seed=42,
        min_alternatives=100,
        max_alternatives=100,
        min_criteria=30,
        max_criteria=30,
    )

    dm4 = decision_matrix(
        seed=42,
        min_alternatives=20,
        max_alternatives=20,
        min_criteria=3,
        max_criteria=3,
    )

    assert dm1.shape == dm2.shape

    assert dm1.shape != dm3.shape and dm1.shape != dm4.shape
    assert dm2.shape != dm3.shape and dm2.shape != dm4.shape


def test_DecisionMatrix_len_vs_shape_ncriteria(decision_matrix):
    dm = decision_matrix(seed=42)

    assert (len(dm), len(dm.criteria)) == np.shape(dm) == dm.shape


def test_DecisionMatrix_self_eq(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )
    same = dm

    assert dm is same
    assert dm.equals(same)


def test_DecisionMatrix_self_ne(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    omtx, oobjectives, oweights, oanames, ocnames = data_values(seed=43)

    other = data.mkdm(
        matrix=omtx,
        objectives=oobjectives,
        weights=oweights,
        alternatives=oanames,
        criteria=ocnames,
    )
    assert not dm.equals(other)


# =============================================================================
# REPR
# =============================================================================


def test_mksm_simple_repr():

    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
    )

    expected = (
        "   C0[\u25bc 0.1] C1[\u25b2 0.2] C2[\u25bc 0.3]\n"
        "A0         1         2         3\n"
        "A1         4         5         6\n"
        "A2         7         8         9\n"
        "[3 Alternatives x 3 Criteria]"
    )

    result = repr(dm)
    assert result == expected


def test_simple_html():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
    )

    expected = PyQuery(
        """
        <div class="decisionmatrix">
            <div>
                <style scoped="">
                    .dataframe tbody tr th:only-of-type {
                        vertical-align: middle;
                    }

                    .dataframe tbody tr th {
                        vertical-align: top;
                    }

                    .dataframe thead th {
                        text-align: right;
                    }
                </style>
                <table border="1" class="dataframe">
                    <thead>
                        <tr style="text-align: right;">
                            <th/>
                            <th>C0[▼ 0.1]</th>
                            <th>C1[▲ 0.2]</th>
                            <th>C2[▼ 0.3]</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th>A0</th>
                            <td>1</td>
                            <td>2</td>
                            <td>3</td>
                        </tr>
                        <tr>
                            <th>A1</th>
                            <td>4</td>
                            <td>5</td>
                            <td>6</td>
                        </tr>
                        <tr>
                            <th>A2</th>
                            <td>7</td>
                            <td>8</td>
                            <td>9</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <em class="decisionmatrix-dim">3 Alternatives x 3 Criteria
            </em>
        </div>
    """
    )

    result = PyQuery(dm._repr_html_())

    assert result.text() == expected.text()


# =============================================================================
# MUST FAIL
# =============================================================================


def test_DecisionMatrix_no_provide_mtx(data_values):
    _, objectives, weights, alternatives, criteria = data_values(seed=42)
    with pytest.raises(TypeError):
        data.mkdm(
            objectives=objectives,
            weights=weights,
            criteria=criteria,
            alternatives=alternatives,
        )


def test_DecisionMatrix_no_provide_objective(data_values):
    mtx, _, weights, alternatives, criteria = data_values(seed=42)
    with pytest.raises(TypeError):
        data.mkdm(
            mtxt=mtx,
            weights=weights,
            criteria=criteria,
            alternatives=alternatives,
        )


def test_DecisionMatrix_invalid_objective(data_values):
    mtx, _, weights, alternatives, criteria = data_values(seed=42)
    objectives = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_weight_no_float(data_values):
    mtx, objectives, _, alternatives, criteria = data_values(seed=42)
    weights = ["hola"]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_missmatch_objective(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)
    objectives = objectives[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_missmatch_dtypes(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
            dtypes=[float],
        )


def test_DecisionMatrix_missmatch_weights(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)
    weights = weights[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_missmatch_anames(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)
    alternatives = alternatives[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_missmatch_cnames(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)
    criteria = criteria[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_mtx_ndim1(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)
    mtx = mtx.flatten()
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


def test_DecisionMatrix_mtx_ndim3(data_values):
    _, objectives, weights, alternatives, criteria = data_values(seed=42)
    mtx = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )


# =============================================================================
# RESULT BASE
# =============================================================================


class test_ResultBase_skacriteria_result_column_no_defined:

    with pytest.raises(TypeError):

        class Foo(data.ResultABC):
            def _validate_result(self, values):
                pass


class test_ResultBase_original_validare_result_fail:
    class Foo(data.ResultABC):
        _skcriteria_result_column = "foo"

        def _validate_result(self, values):
            return super()._validate_result(values)

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

    result = data.RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert np.all(result.method == method)
    assert np.all(result.alternatives == alternatives)
    assert np.all(result.rank_ == rank)
    assert np.all(result.extra_ == result.e_ == extra)


@pytest.mark.parametrize("rank", [[1, 2, 5], [1, 1, 1], [1, 2, 2], [1, 2]])
def test_RankResult_invalid_rank(rank):
    method = "foo"
    alternatives = ["a", "b", "c"]
    extra = {"alfa": 1}

    with pytest.raises(ValueError):
        data.RankResult(
            method=method, alternatives=alternatives, values=rank, extra=extra
        )


def test_RankResult_shape():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    alternatives = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = data.RankResult(
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

    result = data.RankResult(
        method=method, alternatives=alternatives, values=rank, extra=extra
    )

    assert len(result) == length


def test_RankResult_repr():
    method = "foo"
    alternatives = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = data.RankResult(
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
        data.RankResult(
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
        data.KernelResult(
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
        data.KernelResult(
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
