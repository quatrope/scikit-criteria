#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.dict_cmp"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


from skcriteria.utils import dict_cmp


# =============================================================================
# The tests
# =============================================================================


def test_dict_allclose():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": {}}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": {}}
    assert dict_cmp.dict_allclose(left, right)


def test_dict_allclose_same_obj():
    dict0 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    assert dict_cmp.dict_allclose(dict0, dict0)


def test_dict_allclose_different_keys():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "d": 1}
    assert dict_cmp.dict_allclose(left, right) is False


def test_dict_allclose_different_types():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1.0}
    assert dict_cmp.dict_allclose(left, right) is False


def test_dict_allclose_rtol():
    left_rtol = {"a": np.array([100.0, 200.0])}
    right_rtol = {"a": np.array([101.0, 202.0])}  # 1% difference
    assert dict_cmp.dict_allclose(left_rtol, right_rtol) is False
    assert dict_cmp.dict_allclose(left_rtol, right_rtol, rtol=1e-2) is True


def test_dict_allclose_atol():
    left_atol = {"a": np.array([0.0, 1.0])}
    right_atol = {"a": np.array([1e-7, 1.0])}
    assert dict_cmp.dict_allclose(left_atol, right_atol) is False
    assert dict_cmp.dict_allclose(left_atol, right_atol, atol=1e-6) is True


def test_dict_allclose_equal_nan():
    left_nan = {"a": np.array([1.0, np.nan])}
    right_nan = {"a": np.array([1.0, np.nan])}
    assert dict_cmp.dict_allclose(left_nan, right_nan) is False
    assert dict_cmp.dict_allclose(left_nan, right_nan, equal_nan=True) is True
