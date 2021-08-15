#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Container object exposing keys as attributes."""


# =============================================================================
# IMPORTS
# =============================================================================

from collections.abc import Mapping

# =============================================================================
# DOC INHERITANCE
# =============================================================================


class Bunch(Mapping):
    """Container object exposing keys as attributes.

    Concept based on the sklearn.utils.Bunch.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> b = SKCBunch("data", {"a": 1, "b": 2})
    >>> b
    data({a, b})
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, name, data):
        self._name = str(name)
        self._data = data

    def __getitem__(self, k):
        """x.__getitem__(y) <==> x[y]."""
        return self._data[k]

    def __getattr__(self, a):
        """x.__getattr__(y) <==> x.y."""
        try:
            return self[a]
        except KeyError:
            raise AttributeError(a)

    def __iter__(self):
        """x.__iter__() <==> iter(x)."""
        return iter(self._data)

    def __len__(self):
        """x.__len__() <==> len(x)."""
        return len(self._data)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        content = repr(set(self._data)) if self._data else "{}"
        return f"{self._name}({content})"

    def __dir__(self):
        """x.__dir__() <==> dir(x)."""
        return super().__dir__() + list(self._data)
