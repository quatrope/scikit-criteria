#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Framework for define the diffenrent parts of the scikit-criteria library.

"""


# =============================================================================
# IMPORTS
# =============================================================================

from unittest import mock

import attr


# =============================================================================
# META
# =============================================================================

__all__ = ["AttributeClass"]


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@attr.s(frozen=True)
class AttrCC:
    configuration = attr.ib()
    initialization = attr.ib()
    original_setattr = attr.ib()
    full_qualname = attr.ib()

    @property
    def frozen(self):
        return self.configuration.get("frozen", False)


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

        if hasattr(cls, "__configuration__"):
            attrs_configuration = dict(cls.__configuration__)
            del cls.__configuration__
        else:
            attrs_configuration = {}

        if hasattr(cls, "__initialization__"):
            attrs_initialization = cls.__initialization__
            del cls.__initialization__
        else:
            attrs_initialization = attr.NOTHING

        original_setattr = cls.__setattr__

        # https://stackoverflow.com/a/2020083
        module = cls.__module__
        full_qualname = (
            cls.__name__  # Avoid reporting __builtin__
            if module is None or module == str.__class__.__module__ else
            f"{module}.{cls.__name__}")

        cls._attrcc = AttrCC(
            configuration=attrs_configuration,
            initialization=attrs_initialization,
            original_setattr=original_setattr,
            full_qualname=full_qualname)

        cls = attr.s(**attrs_configuration)(cls)

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
                  eq=True, hash=None, init=True, convert=None, metadata=None,
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
            default=default, validator=validator, repr=repr, eq=eq,
            hash=hash, init=init, metadata=metadata, type=type,
            converter=converter, factory=factory, kw_only=kw_only)

        return attrib

    def __attrs_post_init__(self):
        if self._attrcc.initialization is not attr.NOTHING:
            if self._attrcc.frozen:
                to_patch = f"{self._attrcc.full_qualname}.__setattr__"
                patch_with = self._attrcc.original_setattr
                with mock.patch(to_patch, patch_with, create=True):
                    self._attrcc.initialization(self)
            else:
                self._attrcc.initialization(self)
