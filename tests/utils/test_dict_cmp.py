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
