#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for linnear programming based on PuLP.

This file contains an abstraction class to manipulate in a more OOP way
the underlining PuLP model

"""


# =============================================================================
# CONSTANTS
# =============================================================================

_INSTANCE_ATTRIBUTE = "_singleton_instance"


# =============================================================================
# CLASS
# =============================================================================


class Singleton:
    """Restricts the instantiation of a class to one "single"
    instance.

    This is useful when exactly one object is needed to coordinate actions
    across the system.

    The singleton design pattern is one of the twenty-three well-known
    "Gang of Four" design patterns that describe how to solve recurring
    design problems to design flexible and reusable object-oriented software.

    In addition this implementation prevent more than one level of inheritance
    and direct instantiation of this class.

    References
    ----------
    :cite:p:`enwiki:1114075000`
    :cite:p:`gamma1995design`


    """

    def __new__(cls, *args, **kwargs):
        if cls is Singleton:
            raise TypeError("Singleton class can't be instantiated")
        if _INSTANCE_ATTRIBUTE not in vars(cls):
            instance = super().__new__(cls, *args, **kwargs)
            setattr(cls, _INSTANCE_ATTRIBUTE, instance)
        return getattr(cls, _INSTANCE_ATTRIBUTE)

    def __init_subclass__(cls) -> None:
        if cls.mro()[1] != Singleton:
            raise TypeError(
                "Only one level of inheritance from Singleton is allowed"
            )
