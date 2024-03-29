#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
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


def test_add_sphinx_deprecated_directive_whitout_titles():
    original = "foo"
    with_directive = deprecate.add_sphinx_deprecated_directive(
        original, reason="yes", version=0.9
    )
    expected = "foo\n\n.. deprecated:: 0.9\n    yes\n"
    assert expected == with_directive


def test_add_sphinx_deprecated_directive_whit_titles():
    original = (
        "foo\n"
        "\n"
        " Introduction\n"
        " ------------\n"
        "\n"
        " something foo del baz ham eggs"
    )

    with_directive = deprecate.add_sphinx_deprecated_directive(
        original, reason="yes", version=0.9
    )
    expected = (
        "foo\n"
        "\n"
        " .. deprecated:: 0.9\n"
        "     yes\n"
        "\n"
        "\n"
        " Introduction\n"
        " ------------\n"
        "\n"
        " something foo del baz ham eggs"
    )

    assert expected == with_directive


def test_warn_once():
    with pytest.deprecated_call():
        deprecate.warn(reason="foo", version=deprecate.ERROR_GE - 0.5)


def test_warn_error():
    with pytest.raises(deprecate.SKCriteriaDeprecationWarning):
        deprecate.warn(reason="foo", version=deprecate.ERROR_GE)


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
