#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.core.data"""


# =============================================================================
# IMPORTS
# =============================================================================

import io
import warnings

import numpy as np

import pandas as pd

import pyquery

import pytest

import skcriteria as skc
from skcriteria.core import data, dominance, plot, stats


# =============================================================================
# HELPER
# =============================================================================


def construct_iobjectives(arr):
    return [data.Objective.from_alias(obj).value for obj in arr]


def construct_objectives(arr):
    return [data.Objective.from_alias(obj) for obj in arr]


# =============================================================================
# AC ACCESSORS
# =============================================================================


def test__ACArray():
    with warnings.catch_warnings():
        # see: https://stackoverflow.com/a/46721064
        warnings.simplefilter(action="ignore", category=FutureWarning)

        content = ["a", "b", "c"]
        mapping = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}

        arr = data._ACArray(content, mapping.__getitem__)

        assert arr["a"] == [1, 2, 3]
        assert arr["b"] == [4, 5, 6]
        assert arr["c"] == [7, 8, 9]

        assert list(arr) == content

        assert dict(arr.items()) == mapping
        assert list(arr.keys()) == list(arr) == content
        assert sorted(list(arr.values())) == sorted(list(mapping.values()))

        with pytest.raises(AttributeError):
            arr[0] = 1

        with pytest.raises(IndexError):
            arr["foo"]


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
        dm.iobjectives, construct_iobjectives(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)
    np.testing.assert_array_equal(dm.dtypes, [np.float64] * len(criteria))

    np.testing.assert_array_equal(
        dm.minwhere, dm.objectives == data.Objective.MIN
    )
    np.testing.assert_array_equal(
        dm.maxwhere, dm.objectives == data.Objective.MAX
    )


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
        dm.iobjectives, construct_iobjectives(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)

    np.testing.assert_array_equal(
        dm.minwhere, dm.objectives == data.Objective.MIN
    )
    np.testing.assert_array_equal(
        dm.maxwhere, dm.objectives == data.Objective.MAX
    )


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
        dm.iobjectives, construct_iobjectives(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)

    np.testing.assert_array_equal(
        dm.minwhere, dm.objectives == data.Objective.MIN
    )
    np.testing.assert_array_equal(
        dm.maxwhere, dm.objectives == data.Objective.MAX
    )


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
        dm.iobjectives, construct_iobjectives(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)

    np.testing.assert_array_equal(
        dm.minwhere, dm.objectives == data.Objective.MIN
    )
    np.testing.assert_array_equal(
        dm.maxwhere, dm.objectives == data.Objective.MAX
    )


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
        dm.iobjectives, construct_iobjectives(objectives)
    )
    np.testing.assert_array_equal(
        dm.objectives, construct_objectives(objectives)
    )
    np.testing.assert_array_equal(dm.weights, weights)
    np.testing.assert_array_equal(dm.alternatives, alternatives)
    np.testing.assert_array_equal(dm.criteria, criteria)

    np.testing.assert_array_equal(
        dm.minwhere, dm.objectives == data.Objective.MIN
    )
    np.testing.assert_array_equal(
        dm.maxwhere, dm.objectives == data.Objective.MAX
    )


# =============================================================================
# PROPERTIES
# =============================================================================


def test_DecisionMatrix_plot(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    assert isinstance(dm.plot, plot.DecisionMatrixPlotter)
    assert dm.plot._dm is dm


def test_DecisionMatrix_stats(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    assert isinstance(dm.stats, stats.DecisionMatrixStatsAccessor)
    assert dm.stats._dm is dm


def test_DecisionMatrix_dominance(decision_matrix):
    dm = decision_matrix(
        seed=42,
        min_alternatives=10,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=3,
    )

    assert isinstance(dm.dominance, dominance.DecisionMatrixDominanceAccessor)
    assert dm.dominance._dm is dm


# =============================================================================
# DECISION MATRIX
# =============================================================================


def test_DecisionMatrix_constant_criteria():
    dm = data.mkdm(
        matrix=np.array([[1, 2], [1, 4]]),
        objectives=[min, max],
    )
    ccriteria = dm.constant_criteria()

    expected = pd.Series(
        [True, False], index=["C0", "C1"], name="constant_criteria"
    )
    pd.testing.assert_series_equal(ccriteria, expected)


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

    with pytest.deprecated_call():
        dm.copy(**dm.to_dict())


def test_DecisionMatrix_replace(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )
    copy = dm.replace(weights=dm.weights + 1)

    assert dm is not copy
    assert not dm.equals(copy)
    pd.testing.assert_series_equal(dm.weights, copy.weights - 1)


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
        "objectives": construct_iobjectives(objectives),
        "weights": weights,
        "alternatives": np.asarray(alternatives),
        "criteria": np.asarray(criteria),
        "dtypes": np.full(len(weights), float),
    }

    result = dm.to_dict()

    cmp = {k: (np.all(result[k] == expected[k])) for k in result.keys()}
    assert np.all(cmp.values())


def test_DecisionMatrix_to_latex(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    latex = dm.to_latex()

    # create the expected table
    df = dm.to_dataframe()
    df.columns = [rf"\textbf{{{col}}}" for col in df.columns]

    expected = df.to_latex(bold_rows=True)

    expected_lines = expected.splitlines()
    expected_lines.insert(6, r"\midrule")
    expected = "\n".join(expected_lines)

    assert latex == expected


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

    with pytest.deprecated_call():
        result = dm.describe()

    assert result.equals(expected)


# =============================================================================
# IO
# =============================================================================


def test_DecisionMatrix_to_dmsy(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    buff = io.StringIO()
    dm.to_dmsy(buff)

    buff.seek(0)
    dm2 = skc.io.read_dmsy(buff)

    assert dm is not dm2
    skc.testing.assert_dmatrix_equals(dm, dm2)


def test_DecisionMatrix_to_dmsy_filepath_or_buffer_None(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    code = dm.to_dmsy()
    buff = io.StringIO(code)

    dm2 = skc.io.read_dmsy(buff)

    assert dm is not dm2
    skc.testing.assert_dmatrix_equals(dm, dm2)


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


def test_DecisionMatrix_diff(data_values):
    mtx, objectives, weights, alternatives, criteria = data_values(seed=42)

    dm = data.mkdm(
        matrix=mtx,
        objectives=objectives,
        weights=weights,
        alternatives=alternatives,
        criteria=criteria,
    )

    result = dm.diff(dm)

    assert result.has_differences is False
    assert result.left_type is data.DecisionMatrix
    assert result.right_type is data.DecisionMatrix
    assert result.different_types is False
    assert result.members_diff == {}

    # compare with another dm

    omtx, oobjectives, oweights, oanames, ocnames = data_values(seed=43)

    other = data.mkdm(
        matrix=omtx,
        objectives=oobjectives,
        weights=oweights,
        alternatives=oanames,
        criteria=ocnames,
    )

    result = dm.diff(other)

    assert result.has_differences
    assert result.left_type is data.DecisionMatrix
    assert result.right_type is data.DecisionMatrix
    assert result.different_types is False
    assert set(result.members_diff) == {
        "shape",
        "alternatives",
        "weights",
        "objectives",
        "matrix",
        "criteria",
    }


# =============================================================================
# SLICES
# =============================================================================


def test_DecisionMatrix__getitem__():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
        alternatives="A B C".split(),
        criteria="X Y Z".split(),
    )
    assert dm["X"].equals(dm[["X"]])

    expected = data.mkdm(
        matrix=[[1, 3], [4, 6], [7, 9]],
        objectives=[min, min],
        weights=[0.1, 0.3],
        alternatives="A B C".split(),
        criteria="X Z".split(),
    )
    assert dm[["X", "Z"]].equals(expected)


def test_DecisionMatrix_loc():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
        alternatives="A B C".split(),
        criteria="X Y Z".split(),
    )
    assert dm.loc.name == "loc"
    assert dm.loc["A"].equals(dm.loc[["A"]])

    expected = data.mkdm(
        matrix=[[1, 2, 3], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
        alternatives="A C".split(),
        criteria="X Y Z".split(),
    )
    assert dm.loc[["A", "C"]].equals(expected)


def test_DecisionMatrix_iloc():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
        alternatives="A B C".split(),
        criteria="X Y Z".split(),
    )
    assert dm.iloc.name == "iloc"
    assert dm.iloc[2].equals(dm.iloc[[2]])

    expected = data.mkdm(
        matrix=[[1, 2, 3], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
        alternatives="A C".split(),
        criteria="X Y Z".split(),
    )

    assert dm.iloc[[0, 2]].equals(expected)


# =============================================================================
# REPR
# =============================================================================


def test_mkdm__get_cow_headers():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
        criteria="A B C".split(),
    )

    expected = ["A[▼ 0.1]", "B[▲ 0.2]", "C[▼ 0.3]"]
    cow_headers = dm._get_cow_headers()
    np.testing.assert_array_equal(cow_headers, expected)

    expected = [
        "B[▲ 0.2]",
        "A[▼ 0.1]",
    ]
    cow_headers = dm._get_cow_headers(only=["B", "A"])
    np.testing.assert_array_equal(cow_headers, expected)


def test_mkdm_simple_repr():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
    )

    expected = (
        "    C0[▼ 0.1]  C1[▲ 0.2]  C2[▼ 0.3]\n"
        "A0          1          2          3\n"
        "A1          4          5          6\n"
        "A2          7          8          9\n"
        "[3 Alternatives x 3 Criteria]"
    )

    result = repr(dm)
    assert result == expected


def test_mkdm_simple_html():
    dm = data.mkdm(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        objectives=[min, max, min],
        weights=[0.1, 0.2, 0.3],
    )

    expected = pyquery.PyQuery(
        """
        <div class="decisionmatrix">
            <div>
                <style scoped=''>
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
            <em class='decisionmatrix-dim'>3 Alternatives x 3 Criteria
            </em>
        </div>
    """
    )

    result = pyquery.PyQuery(dm._repr_html_())

    assert result.remove("style").text() == expected.remove("style").text()


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
