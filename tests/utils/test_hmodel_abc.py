#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.hmodel_abc.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import pytest

from skcriteria.utils import hmodel_abc

# =============================================================================
# tests
# =============================================================================


def test_hparam():
    hp = hmodel_abc.hparam(1)

    assert hp.metadata[hmodel_abc.HMODEL_METADATA_KEY]["hparam"]


def test_mproperty():
    mp = hmodel_abc.mproperty()
    assert mp.metadata[hmodel_abc.HMODEL_METADATA_KEY]["mproperty"]


def test_create_HModel():
    class MyBase(hmodel_abc.HModelABC):
        v = hmodel_abc.hparam(42)

    class MyModel(MyBase):

        p = hmodel_abc.hparam(25)
        m = hmodel_abc.mproperty()

        @m.default
        def _md(self):
            return self.p + 1

    model = MyModel()
    assert model.v == 42
    assert model.p == 25
    assert model.m == 26

    assert repr(model) == "MyModel(p=25, v=42)"

    model = MyModel(p=27, v=43)
    assert model.v == 43
    assert model.p == 27
    assert model.m == 28
    assert repr(model) == "MyModel(p=27, v=43)"

    with pytest.raises(TypeError):
        MyModel(27)

    with pytest.raises(TypeError):
        MyModel(m=27)
