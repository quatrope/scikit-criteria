#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.deprecate

"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria.utils import deprecate


# =============================================================================
# TEST CLASSES
# =============================================================================


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

    expected_doc = """Zaraza.

        Foo

        .. deprecated:: 0.66
            because foo


        Parameters
        ----------
        a: int
            coso.

        Returns
        -------
        None:
            Nothing to return.

        """

    decorator = deprecate.deprecated(reason="because foo", version=0.66)
    func = decorator(func)

    with pytest.deprecated_call():
        func()

    assert func.__doc__ == expected_doc

    # this can be useful to catch bugs
    # print("-" * 100)
    # print(repr(func.__doc__))
    # print(repr(expected_doc))
    # print("-" * 100)


def test_will_change():
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

    expected_doc = """Zaraza.

        Foo

        .. deprecated:: 0.66
            because foo


        Parameters
        ----------
        a: int
            coso.

        Returns
        -------
        None:
            Nothing to return.

        """

    decorator = deprecate.will_change(reason="because foo", version=0.66)
    func = decorator(func)

    with pytest.warns(deprecate.SKCriteriaFutureWarning):
        func()

    assert func.__doc__ == expected_doc

    # this can be useful to catch bugs
    # print("-" * 100)
    # print(repr(func.__doc__))
    # print(repr(expected_doc))
    # print("-" * 100)
