#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.unames"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria.utils import unames


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_unique_names():
    names, elements = ["foo", "faa"], [0, 1]
    result = dict(unames.unique_names(names=names, elements=elements))
    expected = {"foo": 0, "faa": 1}
    assert result == expected


def test_unique_names_with_duplticates():
    names, elements = ["foo", "foo"], [0, 1]
    result = dict(unames.unique_names(names=names, elements=elements))
    expected = {"foo_1": 0, "foo_2": 1}
    assert result == expected


def test_unique_names_with_different_len():
    names, elements = ["foo", "foo"], [0]
    with pytest.raises(ValueError):
        unames.unique_names(names=names, elements=elements)
