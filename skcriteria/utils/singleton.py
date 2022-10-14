#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# The thread safe part is inspired from:
# https://refactoring.guru/es/design-patterns/singleton/python/example

# =============================================================================
# DOCS
# =============================================================================

"""Singleton base class."""

# =============================================================================
# IMPORTS
# =============================================================================

from threading import Lock


# =============================================================================
# CONSTANTS
# =============================================================================

_INSTANCE_ATTRIBUTE = "_singleton_instance"


# =============================================================================
# CLASS
# =============================================================================


class Singleton:
    """Restricts the instantiation of a class to one "single" instance.

    This is useful when exactly one object is needed to coordinate actions
    across the system.

    The singleton design pattern is one of the twenty-three well-known
    "Gang of Four" design patterns that describe how to solve recurring
    design problems to design flexible and reusable object-oriented software.

    In addition this implementation:

    - Thread safe.
    - Prevent more than one level of inheritance.
    - Direct instantiation of this class.
    - In case of multiple-inheritance, singleton can only be inherited in the
      first place.

    References
    ----------
    :cite:p:`enwiki:1114075000`
    :cite:p:`gamma1995design`

    """

    # lock object that will be used to synchronize thread
    _lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        """If called for the first time, it creates, stores and returns an \
        object. The second time onwards it only returns the stored object.

        This method also prevents Singleton from being instantiated without
        being inherited.

        See help(type) for accurate signature.

        """
        if cls is Singleton:
            raise TypeError("Singleton class can't be instantiated")

        with cls._lock:
            if _INSTANCE_ATTRIBUTE not in vars(cls):
                instance = super().__new__(cls, *args, **kwargs)
                setattr(cls, _INSTANCE_ATTRIBUTE, instance)
        return getattr(cls, _INSTANCE_ATTRIBUTE)

    def __init_subclass__(cls) -> None:
        """Called when a class is subclassed.

        This implementation prevents a Singleton subclass object from being
        inherited again.

        In case of inheritance-multiple, singleton can only be inherited in
        the first place.

        """
        if cls.mro()[1] != Singleton:
            raise TypeError(
                "Only one level of inheritance from Singleton is allowed"
            )
