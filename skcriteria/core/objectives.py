#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Definition of the objectives (MIN, MAX) for the criteria."""


# =============================================================================
# IMPORTS
# =============================================================================

import enum

import numpy as np

from ..utils import deprecated


# =============================================================================
# CONSTANTS
# =============================================================================


class Objective(enum.Enum):
    """Representation of criteria objectives (Minimize, Maximize)."""

    #: Internal representation of minimize criteria
    MIN = -1

    #: Internal representation of maximize criteria
    MAX = 1

    # INTERNALS ===============================================================

    _MIN_STR = "\u25bc"  # ▼
    _MAX_STR = "\u25b2"  # ▲

    #: Another way to name the maximization criteria.
    _MAX_ALIASES = frozenset(
        [
            MAX,
            _MAX_STR,
            max,
            np.max,
            np.nanmax,
            np.amax,
            "max",
            "maximize",
            "+",
            ">",
        ]
    )

    #: Another ways to name the minimization criteria.
    _MIN_ALIASES = frozenset(
        [
            MIN,
            _MIN_STR,
            min,
            np.min,
            np.nanmin,
            np.amin,
            "min",
            "minimize",
            "-",
            "<",
        ]
    )

    # CUSTOM CONSTRUCTOR ======================================================

    @classmethod
    def from_alias(cls, alias):
        """Return a n objective instase based on some given alias."""
        if isinstance(alias, cls):
            return alias
        if isinstance(alias, str):
            alias = alias.lower()
        if alias in cls._MAX_ALIASES.value:
            return cls.MAX
        if alias in cls._MIN_ALIASES.value:
            return cls.MIN
        raise ValueError(f"Invalid criteria objective {alias}")

    # METHODS =================================================================

    def __str__(self):
        """Convert the objective to an string."""
        return self.name

    def to_symbol(self):
        """Return the printable symbol representation of the objective."""
        if self.value in Objective._MIN_ALIASES.value:
            return Objective._MIN_STR.value
        if self.value in Objective._MAX_ALIASES.value:
            return Objective._MAX_STR.value

    # DEPRECATED ==============================================================

    @classmethod
    @deprecated(reason="Use ``Objective.from_alias()`` instead.", version=0.8)
    def construct_from_alias(cls, alias):
        """Return an objective instance based on some given alias."""
        return cls.from_alias(alias)

    @deprecated(reason="Use ``MAX/MIN.to_symbol()`` instead.", version=0.8)
    def to_string(self):
        """Return the printable representation of the objective."""
        return self.to_symbol()
