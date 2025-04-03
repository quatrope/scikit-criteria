#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""On-demand importer of modules.

The on-demand importer is a function that returns a callable object that
imports the module when the object is called.


Notes
-----

This ondemand importer is based on the one from scikit-learn, but adds a
more power to introspection.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import dataclasses as dc
import importlib
import pkgutil
import sys
import types

# =============================================================================
# UTILS
# =============================================================================


def is_package(obj):
    """Check if the object is a package.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is a package, False otherwise.

    """
    return isinstance(obj, types.ModuleType) and hasattr(obj, "__path__")


# =============================================================================
# API
# =============================================================================


@dc.dataclass(frozen=True)
class _OnDemandImporter:
    """
    Enhanced on-demand importer for lazy loading of package modules.

    This class implements a mechanism for lazy loading of modules within a
    package. It postpones the import of a module until it is explicitly
    requested, allowing for more efficient loading of large packages. Unlike
    simpler implementations, this version also provides directory listing
    capabilities.

    Parameters
    ----------
    package_name : str
        The fully qualified name of the package.
    package : types.ModuleType
        The package module object.

    Raises
    ------
    ValueError
        If the provided package object is not actually a package.

    Notes
    -----
    This implementation uses a frozen dataclass to ensure immutability of the
    importer's state.
    """

    package_name: str
    package: types.ModuleType

    def __dataclass_post_init__(self):
        """
        Post-initialization validation.

        Verifies that the provided object is actually a package.

        Raises
        ------
        ValueError
            If the provided package object is not actually a package.
        """
        if not is_package(self.package):
            raise ValueError(f"Object {self.package_name} is not a package")

    @property
    def package_context(self):
        """
        Get the package's context dictionary.

        Returns
        -------
        dict
            Dictionary of the package's variables and modules.
        """
        return vars(self.package)

    @property
    def package_path(self):
        """
        Get the package's search path.

        Returns
        -------
        list
            List of directories where the package's modules can be found.
        """
        return self.package.__path__

    def import_module(self, name):
        """Dynamically imports a submodule and adds it to the parent package \
        context.

        Core of the lazy-loading mechanism that handles imports when attributes
        are requested but not yet loaded. Configures imported packages for
        recursive lazy-loading by replacing their __getattr__ methods.

        Parameters
        ----------
        name : str
            Module name to import (without parent package prefix)

        Returns
        -------
        module
            The imported module or subpackage

        Raises
        ------
        AttributeError
            If module doesn't exist or can't be imported

        Notes
        -----
        - Uses ImportError/AttributeError specifically for Jedi compatibility
        - Jedi's autocompletion engine ignores only these exceptions during
          namespace exploration
        - Caches imported modules in package_context dictionary
        - Sets up recursive importers for subpackages
        - Provides standard attribute error messages

        Implementation handles Jedi's aggressive autocompletion which uses
        __getattr__ to discover available attributes, ignoring only ImportError
        and AttributeError when exploring namespaces.

        The Jupyter ecosystem, which has become the most important interactive
        environment in the scientific domain, uses Jedi as its autocompletion
        engine.

        """

        # Name of the module to import
        to_import_name = f"{self.package_name}.{name}"

        # If the module is already in the context, return it
        try:
            mod_or_pkg = importlib.import_module(to_import_name)
            # Add the module to the context
            self.package_context[name] = mod_or_pkg

            # If the imported module is a package, replace its __getattr__ method
            if is_package(mod_or_pkg):
                ondemand_importer = _OnDemandImporter(
                    to_import_name, mod_or_pkg
                )
                mod_or_pkg.__getattr__ = ondemand_importer.import_module
        except ImportError:
            try:
                mod_or_pkg = self.package_context[name]
            except KeyError:
                # If the module is not in the context, raise an error
                raise AttributeError(
                    f"Module '{self.package_name}' has no attribute '{name}'"
                )

        return mod_or_pkg

    def list_available_modules(self):
        """List all available modules in the package.

        This method combines the already imported modules with the modules
        available on disk that have not yet been imported.

        Returns
        -------
        list
            Sorted list of all available module names in the package.

        """
        available_modules = set(self.package_context)
        available_modules.update(
            name for _, name, _ in pkgutil.iter_modules(self.package_path)
        )
        return sorted(available_modules)


def mk_ondemand_importer_for(package_name):
    """Create an on-demand importer for a specific package.

    This function creates and returns an instance of _OnDemandImporter for the
    specified package. The package must already be imported and available in
    sys.modules.

    Parameters
    ----------
    package_name : str
        The fully qualified name of the package for which to create an
        importer.

    Returns
    -------
    _OnDemandImporter
        An instance of _OnDemandImporter configured for the specified package.

    Examples
    --------
    >>> # In a package's __init__.py
    >>> importer = mk_ondemand_importer_for(__name__)
    >>> __getattr__ = importer.import_module
    >>> __dir__ = importer.list_available_modules

    """
    package = sys.modules[package_name]
    ondemand_importer = _OnDemandImporter(package_name, package)
    return ondemand_importer
