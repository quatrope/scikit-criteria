#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

import inspect
import sys
import types
from unittest import mock

import pytest

import skcriteria as skc
from skcriteria.utils import ondemand_import

# =============================================================================
# HELPERS FUNCTIONS
# =============================================================================


def is_getattr_on_demand(func):
    return inspect.ismethod(func) and (
        func.__func__
        is ondemand_import.OnDemandImporter.import_or_get_attribute
    )


def is_dir_on_demand(func):
    return inspect.ismethod(func) and (
        func.__func__
        is ondemand_import.OnDemandImporter.list_available_modules
    )


# for mocking
class FakeModule(types.ModuleType):
    pass


# =============================================================================
# TESTS
# =============================================================================


def test_is_package():

    # a package
    assert ondemand_import.is_package(skc)

    # a module
    assert not ondemand_import.is_package(ondemand_import)


def test_OnDemandImporter():

    # on_demand_importer_for_top_level
    fake_package = FakeModule("foo_pkg")
    fake_package.__path__ = "foo_path"  # this make a module a package
    fake_package.some_attribute = "value"

    odimporter = ondemand_import.OnDemandImporter("foo_pkg", fake_package)

    assert odimporter.package_name == "foo_pkg"
    assert odimporter.package is fake_package
    assert odimporter.package_path == "foo_path"
    assert odimporter.import_or_get_attribute("some_attribute") == "value"
    assert "__path__" in odimporter.package_context

    with pytest.raises(AttributeError, match="no_exists"):
        odimporter.import_or_get_attribute("no_exists")

    with mock.patch("pkgutil.iter_modules", return_value=[("", "extra", 1)]):
        content = odimporter.list_available_modules()

    assert content == sorted(list(odimporter.package_context) + ["extra"])

    # subpackage
    fake_subpackage = FakeModule("foo_pkg.foo_sub_pkg")
    fake_subpackage.__path__ = "foo_sub_path"  # this make a module a package

    with mock.patch("importlib.import_module", return_value=fake_subpackage):
        value = odimporter.import_or_get_attribute("foo_sub_pkg")

    assert value is fake_subpackage
    assert is_getattr_on_demand(value.__getattr__)
    assert is_dir_on_demand(value.__dir__)

    # module
    fake_module = FakeModule("foo_pkg.foo_module")

    with mock.patch("importlib.import_module", return_value=fake_module):
        value = odimporter.import_or_get_attribute("foo_module")

    assert value is fake_module
    assert not (
        hasattr(value, "__getattr__")
        and is_getattr_on_demand(value.__getattr__)
    )
    assert not (hasattr(value, "__dir__") and is_dir_on_demand(value.__dir__))


def test_OnDemandImporter_for_no_package():
    with pytest.raises(ValueError, match="Object 'foo_pkg' is not a package"):
        ondemand_import.OnDemandImporter("foo_pkg", None)


def test_mk_ondemand_importer_for():

    fake_package = FakeModule("foo_pkg")
    fake_package.__path__ = "foo_path"  # this make a module a package
    fake_package.some_attribute = "value"

    with mock.patch.dict(sys.modules, {"foo_pkg": fake_package}):
        ondemand_importer = ondemand_import.mk_ondemand_importer_for("foo_pkg")

    assert ondemand_importer.package_name == "foo_pkg"
    assert ondemand_importer.package is fake_package
    assert ondemand_importer.package_path == "foo_path"
    assert (
        ondemand_importer.import_or_get_attribute("some_attribute") == "value"
    )
    assert "__path__" in ondemand_importer.package_context
