#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.npdict_cmp

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


from skcriteria.utils import npdict_cmp


# =============================================================================
# The tests
# =============================================================================


def test_npdict_all_equals_():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    assert npdict_cmp.npdict_all_equals(left, right)


def test_npdict_all_equals_same_obj():
    dict0 = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    assert npdict_cmp.npdict_all_equals(dict0, dict0)


def test_npdict_all_equals_different_keys():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "d": 1}
    assert npdict_cmp.npdict_all_equals(left, right) is False


def test_npdict_all_equals_different_types():
    left = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1}
    right = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6]), "c": 1.0}
    assert npdict_cmp.npdict_all_equals(left, right) is False
