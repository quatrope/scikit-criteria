#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Module containing the basic functionality
for the data representation used inside Scikit-Criteria.

"""


# =============================================================================
# META
# =============================================================================

__all__ = ['Data', 'DataValidationError', 'ascriteria']


# =============================================================================
# IMPORTS
# =============================================================================

import copy
import itertools as it

import numpy as np

from .attribute import AttributeClass

from .serializer import DataSerializerProxy
# from .display.plot import DataPlotProxy


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
        Criteria array as intergers (-1 for minimize, 1 for maximize).

    """
    pcriteria = np.empty(len(criteria))
    for idx, crit in enumerate(criteria):
        if crit in ALIASES:
            pcriteria[idx] = ALIASES[crit]
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
    serializer = AttributeClass.parameter(init=False, repr=False)

    __configuration__ = {
        "repr": False, "frozen": True,
        "order": False, "eq": False}

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

        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if len(self.weights) != len(self.criteria):
                raise DataValidationError(
                    f"{len(self.weights)} weights given "
                    f"for {len(self.criteria)} criteria")

        if self.anames is None:
            self.anames = tuple(
                f"A{idx}" for idx in range(self.mtx.shape[0]))
        else:
            self.anames = tuple(self.anames)
            if len(self.anames) != self.mtx.shape[0]:
                raise DataValidationError(
                    f"{len(self.anames)} names given "
                    f"for {self.mtx.shape[0]} alternatives")

        if self.cnames is None:
            self.cnames = tuple(
                f"C{idx}" for idx in range(self.mtx.shape[1]))
        else:
            self.cnames = tuple(self.cnames)
            if len(self.cnames) != self.mtx.shape[1]:
                raise DataValidationError(
                    f"{len(self.cnames)} names given "
                    f"for {self.mtx.shape[1]} criteria")

        self.serializer = DataSerializerProxy(self)
        self.plot = None  # DataPlotProxy(self)

    def __eq__(self, other):
        if not isinstance(other, Data):
            return NotImplemented
        return self is other or (
            np.all(self.mtx == other.mtx) and
            np.all(self.criteria == other.criteria) and
            np.all(self.weights == other.weights) and
            self.anames == other.anames and
            self.cnames == other.cnames)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return self.serializer.to_text()

    def _repr_html_(self):
        return self.serializer.to_html()

    def copy(self):
        """Create a deep copy of the Data object.

        """
        return Data(
            mtx=self.mtx.copy(),
            criteria=self.criteria.copy(),
            weights=self.weights.copy(),
            anames=copy.copy(self.anames),
            cnames=copy.copy(self.cnames))
