#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.extend"""


# =============================================================================
# IMPORTS
# =============================================================================

import io
import pathlib
from unittest import mock

import numpy as np

import pytest

import skcriteria as skc


import yaml


# =============================================================================
# YAML TEST
# =============================================================================


@pytest.mark.parametrize("iterable", [set, frozenset])
def test_CustomYAMLDumper_iterable_sets(iterable):
    data = iterable([1, 2, 3])
    out = yaml.dump([data], Dumper=skc.io.dmsy.CustomYAMLDumper)
    assert out == "[[1, 2, 3]]\n"


@pytest.mark.parametrize("iterable", [list, tuple, np.array])
def test_CustomYAMLDumper_iterable_ordered(iterable):
    data = iterable([1, 2, 3])
    out = yaml.dump([data], Dumper=skc.io.dmsy.CustomYAMLDumper)
    assert out == "- [1, 2, 3]\n"


def test_CustomYAMLDumper_scalar():
    for dtype in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]:
        data = dtype(1)
        out = yaml.dump(data, Dumper=skc.io.dmsy.CustomYAMLDumper)
        assert out == "1\n...\n"

    for dtype in [
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    ]:
        data = dtype(1.0)
        out = yaml.dump(data, Dumper=skc.io.dmsy.CustomYAMLDumper)
        assert out == "1.0\n...\n"

    out = yaml.dump(np.bool_(True), Dumper=skc.io.dmsy.CustomYAMLDumper)
    assert out == "true\n...\n"


# =============================================================================
# IO TEST
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("alternatives", [1, 10, 100])
@pytest.mark.parametrize("criteria", [1, 10, 100])
def test_read_write_dmsy_buffer(decision_matrix, alternatives, criteria):
    dm = decision_matrix(
        max_criteria=criteria,
        min_criteria=criteria,
        max_alternatives=alternatives,
        min_alternatives=alternatives,
    )

    buff = io.StringIO()
    skc.io.to_dmsy(dm, buff)
    buff.seek(0)
    dm2 = skc.io.read_dmsy(buff)

    assert dm is not dm2
    skc.testing.assert_dmatrix_equals(dm, dm2)


@pytest.mark.slow
@pytest.mark.parametrize("path", [pathlib.Path("test.dmsy"), "test.dmsy"])
@pytest.mark.parametrize("alternatives", [1, 10, 100])
@pytest.mark.parametrize("criteria", [1, 10, 100])
def test_read_write_dmsy_path(decision_matrix, path, alternatives, criteria):
    dm = decision_matrix(
        max_criteria=criteria,
        min_criteria=criteria,
        max_alternatives=alternatives,
        min_alternatives=alternatives,
    )

    mock_open = mock.mock_open()
    with mock.patch("builtins.open", mock_open):
        skc.io.to_dmsy(dm, path)

    mock_open.assert_called_once_with(path, "w")

    mock_file = mock_open()
    mock_file.write.assert_called()

    content = "".join([e[0][0] for e in mock_file.write.call_args_list])
    mock_open = mock.mock_open(read_data=content)
    with mock.patch("builtins.open", mock_open):
        dm2 = skc.io.read_dmsy(path)

    mock_open.assert_called_once_with(path, "r")

    assert dm is not dm2
    skc.testing.assert_dmatrix_equals(dm, dm2)


def test_DMSY_YAML_format():
    dm = skc.mkdm(
        matrix=[[1, 2], [3, 4]],
        objectives=[1, -1],
        weights=[1, 2],
        alternatives=["A", "B"],
        criteria=["C", "D"],
        dtypes=[float, float],
    )

    buff = io.StringIO()
    skc.io.to_dmsy(dm, buff)

    dm_data_expected = (
        "data:\n"
        "  matrix:\n"
        "  - [1.0, 2.0]\n"
        "  - [3.0, 4.0]\n"
        "  objectives: [1, -1]\n"
        "  weights: [1.0, 2.0]\n"
        "  dtypes: [float64, float64]\n"
        "  alternatives: [A, B]\n"
        "  criteria: [C, D]\n"
        "  extra: {}\n"
    )

    result = buff.getvalue()
    dm_data = result[result.index("data:\n"):]  # fmt: skip

    assert dm_data == dm_data_expected
