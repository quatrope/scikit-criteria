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

import string

import numpy as np

import pytest

from skcriteria.utils import decorators


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_doc_inherit():

    chars = tuple(string.ascii_letters + string.digits)
    random = np.random.default_rng(seed=42)

    doc = "".join(random.choice(chars, 1000))

    def func_a():
        ...

    func_a.__doc__ = doc

    @decorators.doc_inherit(func_a)
    def func_b():
        ...

    @decorators.doc_inherit(doc)
    def func_c():
        ...

    assert doc == func_a.__doc__ == func_b.__doc__ == func_c.__doc__


def test_deprecated():
    def func():
        """Zaraza.

        Foo

        Parameters
        ----------
        a: int
            coso.

        Returns
        -------
        None:
            Nothing to return.

        """
        pass

    expected_doc = func.__doc__

    decorator = decorators.deprecated(reason="because foo", version=0.66)
    func = decorator(func)

    with pytest.deprecated_call():
        func()

    assert func.__doc__ == expected_doc

    # this can be useful to catch bugs
    # print("-" * 100)
    # print(repr(func.__doc__))
    # print(repr(expected_doc))
    # print("-" * 100)
