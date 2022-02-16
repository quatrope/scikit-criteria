#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Accessor base class."""

# =============================================================================
# ACESSOR ABC
# =============================================================================


class AccessorABC:

    _DEFAULT_KIND = None

    def __init_subclass__(cls) -> None:
        if cls._DEFAULT_KIND is None:
            raise TypeError(f"{cls!r} must define a _DEFAULT_KIND")

    def __call__(self, kind=None, **kwargs):
        """"""
        kind = self._DEFAULT_KIND if kind is None else kind

        if kind.startswith("_"):
            raise ValueError(f"invalid kind name '{kind}'")

        method = getattr(self, kind, None)
        if not callable(method):
            raise ValueError(f"Invalid kind name '{kind}'")

        return method(**kwargs)
