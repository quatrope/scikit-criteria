#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.decorator"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

import pytest

from skcriteria.utils import AccessorABC


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_AccessorABC():
    class FooAccessor(AccessorABC):
        _default_kind = "zaraza"

        def __init__(self, v):
            self._v = v

        def zaraza(self):
            return self._v

    acc = FooAccessor(np.random.random())
    assert acc("zaraza") == acc.zaraza() == acc()


def test_AccessorABC_no__default_kind():
    with pytest.raises(TypeError):

        class FooAccessor(AccessorABC):
            pass

    with pytest.raises(TypeError):
        AccessorABC()


def test_AccessorABC_invalid_kind():
    class FooAccessor(AccessorABC):
        _default_kind = "zaraza"

        def __init__(self):
            self.dont_work = None

        def _zaraza(self):
            pass

    acc = FooAccessor()

    with pytest.raises(ValueError):
        acc("_zaraza")

    with pytest.raises(ValueError):
        acc("dont_work")
