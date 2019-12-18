
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Module containing the basic functionality
for the data representation used inside Scikit-Criteria.

"""

__all__ = ['Data']


# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

import numpy as np

from .attribute import AttributeClass


# =============================================================================
# CONSTANTS
# =============================================================================

MIN = -1
"""Int: Minimization criteria"""

MIN_ALIASES = [MIN, min, np.min, np.nanmin, np.amin, "min", "minimize"]
"""Another ways to name the minimization criteria."""

MAX = 1
"""Int: Maximization criteria"""

MAX_ALIASES = [MAX, max, np.max, np.nanmax, np.amax, "max", "maximize"]
"""Another way to name the maximization criteria."""


CRITERIA_STR = {
    MIN: "min",
    MAX: "max"
}

TABULATE_PARAMS = {
    "headers": "firstrow",
    "numalign": "center",
    "stralign": "center",
}


ALIASES = dict(it.chain(
    dict.fromkeys(MIN_ALIASES, MIN).items(),
    dict.fromkeys(MAX_ALIASES, MAX).items()))


# =============================================================================
# EXCEPTION
# =============================================================================

class DataValidationError(ValueError):
    """Raises when the data is insconsistent"""


# =============================================================================
# FUNCTIONS
# =============================================================================

def ascriteria(criteria):
    """Validate and convert a criteria array

    Check if the iterable only contains MIN (or any alias) and MAX
    (or any alias) values. And also always returns an ndarray representation
    of the iterable.

    Parameters
    ----------
    criteria : Array-like
        Iterable containing all the values to be validated by the function.

    Returns
    -------
    numpy.ndarray :
        Criteria array.

    """
    pcriteria = np.empty(len(criteria))
    for idx, c in enumerate(criteria):
        if c not in ALIASES:
            pcriteria[idx] = ALIASES[c]
        else:
            raise DataValidationError(
                "Criteria Array only accept minimize or maximize Values. "
                f"Found {criteria}")
    return pcriteria


# =============================================================================
# DATACLASS
# =============================================================================


class Data(AttributeClass):
    """Multi-Criteria data representation.

    This make easy to manipulate:

    - The matrix of alternatives. (``mtx``)
    - The array with the sense of optimality of every criteria (``criteria``).
    - Optional weights of the criteria (``weights``)
    - Optional names of the alternatives (``anames``) and the
      criteria (``cnames``)
    - Optional metadata (``meta``)

    """

    mtx = AttributeClass.parameter()
    criteria = AttributeClass.parameter()
    weights = AttributeClass.parameter(default=None)
    anames = AttributeClass.parameter(default=None)
    cnames = AttributeClass.parameter(default=None)
    plot = AttributeClass.parameter(init=False, repr=False)

    def __initialization__(self):
        self.mtx = np.asarray(self.mtx)
        if np.ndim(self.mtx) != 2:
            raise DataValidationError(
                f"'mtx' must have 2 dimensions. Found {np.ndim(self.mtx)}")

        self.criteria = ascriteria(self.criteria)
        if len(self.criteria) != np.shape(self.mtx)[1]:
            raise DataValidationError(
                f"{len(self.criteria)} senses of optimality given "
                f"but mtx has {self.mtx.shape[1]} criteria")

        if self.weights is None:
            self.weights = np.ones(self.criteria.shape, dtype=float)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
            if len(self.weights) != len(self.criteria):
                raise DataValidationError(
                    f"{len(self.weights)} weights given "
                    f"for {len(self.criteria)} criteria")

        if self.anames is None
            self.anames = tuple(
                f"A{idx}" for idx in range(mtx.shape[0]))
        else:
            self.anames = tuple(self.anames)
            if len(self.anames) !=  self.mtx.shape[0]:
                raise ValueError(
                    f"{len(self.anames)} names given "
                    f"for {self.mtx.shape[0]} alternatives")

        if self.cnames is None
            self.cnames = tuple(
                f"C{idx}" for idx in range(mtx.shape[1]))
        else:
            self.cnames = tuple(self.cnames)
            if len(self.cnames) !=  self.mtx.shape[1]:
                raise ValueError(
                    f"{len(self.cnames)} names given "
                    f"for {self.mtx.shape[1]} criteria")

        self.plot = None

