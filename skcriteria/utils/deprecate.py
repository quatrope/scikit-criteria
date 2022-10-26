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

from deprecated import deprecated as _deprecated


# =============================================================================
# CONSTANTS
# =============================================================================

# _ If the version of the warning is >= ERROR_GE the action is setted to
# 'error', otherwise is 'once'.
ERROR_GE = 1.0

# =============================================================================
# WARNINGS
# =============================================================================


class SKCriteriaDeprecationWarning(DeprecationWarning):
    """Skcriteria deprecation warning."""


class SKCriteriaFutureWarning(FutureWarning):
    """Skcriteria future warning."""


# =============================================================================
# FUNCTIONS
# =============================================================================

DEPRECATION_DIRECTIVE = """
{indent}.. deprecated:: {version}
{indent}    {reason}
"""


def _create_doc_with_deprecated_directive(text, *, reason, version):
    # first let split the text in lines
    lines = text.splitlines()

    # the location is where in between lines we must insert the
    # deprecation directive. By default "at the end"
    location = len(lines)

    # indentation is how much away from the margin (in number os spaces) we
    # must insert the directive. By default n indentation is required
    indentation = ""

    # we iterate line by line
    for idx, line in enumerate(lines):

        line_stripped = line.strip()

        # if we found a line full of "-" is a underline of the first section
        # in numpy format.
        # check: https://numpydoc.readthedocs.io/en/latest/format.html
        if line_stripped and line_stripped.replace("-", "") == "":

            # the the location of the directive is one line above the first
            # section
            location = idx - 2

            # and the indentation is the number os white spaces on the left
            indentation = " " * (len(line) - len(line.lstrip()))

            break

    # we create the directive here
    directive = DEPRECATION_DIRECTIVE.format(
        reason=reason, version=version, indent=indentation
    )

    # we insert the directive in the correct location
    lines.insert(location, directive)

    # recreate the doct with the directive
    new_doc = "\n".join(lines)
    return new_doc


# =============================================================================
# DECORATORS
# =============================================================================


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
    add_warning = _deprecated(
        reason=reason,
        version=version,
        category=SKCriteriaDeprecationWarning,
        action=("error" if version >= ERROR_GE else "once"),
    )

    def _dec(func):
        decorated_func = add_warning(func)
        decorated_func.__doc__ = _create_doc_with_deprecated_directive(
            func.__doc__, reason=reason, version=version
        )
        return decorated_func

    return _dec


def will_change(*, reason, version):
    """Mark functions, classes and methods as "to be changed".

    It will result in a warning being emitted when the object is called,
    and the "deprecated" directive was added to the docstring.

    Parameters
    ----------
    reason: str
        Reason message which documents the "to be changed" in your library.
    version: str
        Version of your project which marks as  this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH".

    Notes
    -----
    This decorator is a thin layer over
    :py:func:`deprecated.deprecated`.

    Check: <github `https://pypi.org/project/Deprecated/`>__

    """
    add_warning = _deprecated(
        reason=reason,
        version=version,
        category=SKCriteriaFutureWarning,
        action="once",
    )

    def _dec(func):
        decorated_func = add_warning(func)
        decorated_func.__doc__ = _create_doc_with_deprecated_directive(
            func.__doc__, reason=reason, version=version
        )
        return decorated_func

    return _dec
