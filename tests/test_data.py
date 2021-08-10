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

from skcriteria import data


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


def test_simple_creation(data_values):

    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)
    np.testing.assert_array_equal(dm.dtypes, [np.float64] * len(cnames))


def test_no_provide_weights(data_values):
    mtx, objectives, _, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        anames=anames,
        cnames=cnames,
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
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_anames(data_values):

    mtx, objectives, weights, _, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        cnames=cnames,
    )

    anames = [f"A{idx}" for idx in range(mtx.shape[0])]

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_cnames(data_values):
    mtx, objectives, weights, anames, _ = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
    )

    cnames = [f"C{idx}" for idx in range(mtx.shape[1])]

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


def test_no_provide_cnames_and_anames(data_values):
    mtx, objectives, weights, _, _ = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
    )

    anames = [f"A{idx}" for idx in range(mtx.shape[0])]
    cnames = [f"C{idx}" for idx in range(mtx.shape[1])]

    np.testing.assert_array_equal(dm.matrix, mtx)
    np.testing.assert_array_equal(
        dm.objectives_values, construct_objectives_values(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.anames, anames)
    np.testing.assert_array_equal(dm.cnames, cnames)


# =============================================================================
# UTILITIES
# =============================================================================


def test_copy(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )
    copy = dm.copy()

    assert dm is not copy
    assert dm.equals(copy)


def test_to_dataframe(data_values):

    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    df = dm.to_dataframe()

    rows = np.vstack((construct_objectives(objectives), weights, mtx))
    expected = pd.DataFrame(
        rows, index=["objectives", "weights"] + anames, columns=cnames
    )

    pd.testing.assert_frame_equal(df, expected)


def test_to_dict(data_values):

    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    expected = {
        "matrix": mtx,
        "objectives": construct_objectives_values(objectives),
        "weights": weights,
        "anames": np.asarray(anames),
        "cnames": np.asarray(cnames),
        "dtypes": np.full(len(weights), float),
    }

    result = dm.to_dict()

    cmp = {k: (np.all(result[k] == expected[k])) for k in result.keys()}
    assert np.all(cmp.values())


def test_describe(data_values):

    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    expected = pd.DataFrame(mtx, columns=cnames, index=anames).describe()

    result = dm.describe()

    assert result.equals(expected)


# =============================================================================
# CMP
# =============================================================================


def test_len(decision_matrix):
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


def test_shape(decision_matrix):
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


def test_len_vs_shape_ncriteria(decision_matrix):
    dm = decision_matrix(seed=42)

    assert (len(dm), len(dm.cnames)) == np.shape(dm) == dm.shape


def test_self_eq(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )
    same = dm

    assert dm is same
    assert dm.equals(same)


def test_self_ne(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        anames=anames,
        cnames=cnames,
    )

    omtx, oobjectives, oweights, oanames, ocnames = data_values(seed=43)

    other = data.mkdm(
        matrix=omtx,
        objectives=oobjectives,
        weights=oweights,
        anames=oanames,
        cnames=ocnames,
    )
    assert not dm.equals(other)


# =============================================================================
# REPR
# =============================================================================


def test_simple_repr():

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


def test_no_provide_mtx(data_values):
    _, objectives, weights, anames, cnames = data_values(seed=42)
    with pytest.raises(TypeError):
        data.mkdm(
            objectives=objectives,
            weights=weights,
            cnames=cnames,
            anames=anames,
        )


def test_no_provide_objective(data_values):
    mtx, _, weights, anames, cnames = data_values(seed=42)
    with pytest.raises(TypeError):
        data.mkdm(mtxt=mtx, weights=weights, cnames=cnames, anames=anames)


def test_invalid_objective(data_values):
    mtx, _, weights, anames, cnames = data_values(seed=42)
    objectives = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_weight_no_float(data_values):
    mtx, objectives, _, anames, cnames = data_values(seed=42)
    weights = ["hola"]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_missmatch_objective(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)
    objectives = objectives[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_missmatch_dtypes(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=[float],
        )


def test_missmatch_weights(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)
    weights = weights[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_missmatch_anames(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)
    anames = anames[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_missmatch_cnames(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)
    cnames = cnames[1:]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_mtx_ndim1(data_values):
    mtx, objectives, weights, anames, cnames = data_values(seed=42)
    mtx = mtx.flatten()
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


def test_mtx_ndim3(data_values):
    _, objectives, weights, anames, cnames = data_values(seed=42)
    mtx = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    with pytest.raises(ValueError):
        data.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )


# =============================================================================
# RANK RESULT
# =============================================================================


def test_RankResult():
    method = "foo"
    anames = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = data.RankResult(
        method=method, anames=anames, rank=rank, extra=extra
    )

    assert np.all(result.method == method)
    assert np.all(result.anames == anames)
    assert np.all(result.rank_ == rank)
    assert np.all(result.extra_ == result.e_ == extra)


@pytest.mark.parametrize("rank", [[1, 2, 5], [1, 1, 1], [1, 2, 2], [1, 2]])
def test_RankResult_invalid_rank(rank):
    method = "foo"
    anames = ["a", "b", "c"]
    extra = {"alfa": 1}

    with pytest.raises(ValueError):
        data.RankResult(method=method, anames=anames, rank=rank, extra=extra)


def test_RankResult_shape():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    anames = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = data.RankResult(
        method=method, anames=anames, rank=rank, extra=extra
    )

    assert result.shape == (length, 1)


def test_RankResult_len():
    random = np.random.default_rng(seed=42)
    length = random.integers(10, 100)

    rank = np.arange(length) + 1
    anames = [f"A.{r}" for r in rank]
    method = "foo"
    extra = {}

    result = data.RankResult(
        method=method, anames=anames, rank=rank, extra=extra
    )

    assert len(result) == length


def test_RankResult_repr():
    method = "foo"
    anames = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = data.RankResult(
        method=method, anames=anames, rank=rank, extra=extra
    )

    expected = "      a  b  c\n" "Rank  1  2  3\n" "[Method: foo]"

    assert repr(result) == expected


def test_RankResult_repr_html():
    method = "foo"
    anames = ["a", "b", "c"]
    rank = [1, 2, 3]
    extra = {"alfa": 1}

    result = PyQuery(
        data.RankResult(
            method=method, anames=anames, rank=rank, extra=extra
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
