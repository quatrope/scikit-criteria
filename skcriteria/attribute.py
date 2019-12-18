#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Framework for define the diffenrent parts of the scikit-criteria library.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import attr


__all__ = ["AttributeClass"]


# =============================================================================
# METACLASS
# =============================================================================

class AttributeMeta(type):
    """This class define the logic of how to create an scikit-criteria class.


    """

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        try:
            cls != AttributeClass
        except NameError:
            return cls

        if not hasattr(cls, "__configuration__"):
            cls.__configuration__ = {}

        if not hasattr(cls, "__initialization__"):
            cls.__initialization__ = attr.NOTHING

        cls = attr.s(**cls.__configuration__)(cls)

        return cls


# =============================================================================
# API
# =============================================================================

class AttributeClass(metaclass=AttributeMeta):
    """Base class for all the scikit-criteria class.

    Currently (in conjuction with the AttributeMeta class) is only
    a thin wrapper over attrs (https://www.attrs.org/).

    """

    @classmethod
    def parameter(cls, default=attr.NOTHING, validator=None, repr=True,
                  cmp=True, hash=None, init=True, convert=None, metadata=None,
                  type=None, converter=None, factory=None, kw_only=False,
                  instanceof=attr.NOTHING):
        """Create a new attribute for the current class"""

        if instanceof is not attr.NOTHING:
            if validator:
                validator = [
                    validator, attr.validators.instance_of(instanceof)]
            else:
                validator = attr.validators.instance_of(instanceof)

        attrib = attr.ib(
            default=default, validator=validator, repr=repr, cmp=cmp,
            hash=hash, init=init, convert=convert, metadata=metadata,
            type=type, converter=converter, factory=factory, kw_only=kw_only)

        return attrib

    def __attrs_post_init__(self):
        if self.__initialization__ is not attr.NOTHING:
            self.__initialization__()
