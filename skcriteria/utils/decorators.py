#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Multiple decorator to use inside scikit-criteria."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from inspect import isclass

from custom_inherit import doc_inherit as _doc_inherit

from deprecated import deprecated as _deprecated

# =============================================================================
# DOC INHERITANCE
# =============================================================================


def doc_inherit(parent, warn_class=True):
    """Inherit the 'parent' docstring.

    Returns a function/method decorator that, given parent, updates
    the docstring of the decorated function/method based on the `numpy`
    style and the corresponding attribute of parent.

    Parameters
    ----------
    parent : Union[str, Any]
        The docstring, or object of which the docstring is utilized as the
        parent docstring during the docstring merge.
    warn_class: bool
        If it is true, and the decorated is a class, it throws a warning
        since there are some issues with inheritance of documentation in
        classes.

    Notes
    -----
    This decorator is a thin layer over
    :py:func:`custom_inherit.doc_inherit decorator`.

    Check: <github `https://github.com/rsokl/custom_inherit`>__


    """

    def _wrapper(obj):
        if isclass(obj) and warn_class:
            warnings.warn(
                f"{obj} is a class, check if the "
                "documentation was inherited properly "
            )
        dec = _doc_inherit(parent, style="numpy")
        return dec(obj)

    return _wrapper


# =============================================================================
# Deprecation
# =============================================================================


class SKCriteriaDeprecationWarning(DeprecationWarning):
    """Skcriteria deprecation warning."""


# _ If the version of the warning is >= ERROR_GE the action is setted to
# 'error', otherwise is 'once'.
ERROR_GE = 1.0


def deprecated(*, reason, version):
    """Mark functions, classes and methods as deprecated.

    It will result in a warning being emitted when the object is called,
    and the "deprecated" directive was added to the docstring.

    Parameters
    ----------
    reason: str
        Reason message which documents the deprecation in your library.
    version: str
        Version of your project which deprecates this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH".

    Notes
    -----
    This decorator is a thin layer over
    :py:func:`deprecated.deprecated`.

    Check: <github `https://pypi.org/project/Deprecated/`>__

    """
    return _deprecated(
        reason=reason,
        version=version,
        category=SKCriteriaDeprecationWarning,
        action=("error" if version >= ERROR_GE else "once"),
    )
