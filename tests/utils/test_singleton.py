#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.decorator

"""


# =============================================================================
# IMPORTS
# =============================================================================


import pytest

from skcriteria.utils import Singleton


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_Singleton():
    class SingleInstance(Singleton):
        ...

    assert SingleInstance() is SingleInstance()


def test_Singleton_cant_be_instantiated():
    with pytest.raises(TypeError):
        Singleton()


def test_Singleton_cant_be_inherited_more_than_one():
    class SingleInstance(Singleton):
        ...

    with pytest.raises(TypeError):

        class MustFail(SingleInstance):
            ...


def test_Singleton_one_instance_by_class():
    class SingleInstance1(Singleton):
        ...

    class SingleInstance2(Singleton):
        ...

    assert SingleInstance1() is SingleInstance1()
    assert isinstance(SingleInstance1(), SingleInstance1)
    assert SingleInstance1() is SingleInstance1()
    assert not isinstance(SingleInstance1(), SingleInstance2)

    assert SingleInstance2() is SingleInstance2()
    assert isinstance(SingleInstance2(), SingleInstance2)
    assert not isinstance(SingleInstance2(), SingleInstance1)

    assert SingleInstance1() is not SingleInstance2()
