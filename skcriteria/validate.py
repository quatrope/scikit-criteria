#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# =============================================================================
# DOCS
# =============================================================================

"""This module core functionalities for validate the data
used inside scikit criteria.

- Constants that represent minimization and mazimization criteria.
- Scikit-Criteria Criteria ndarray creation.
- Scikit-Criteria Data validation.

"""

__all__ = [
    'MIN', 'MAX',
    'DataValidationError',
    'criteriarr',
    'validate_data']


# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

import numpy as np


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


ALIASES = dict(it.chain(dict.fromkeys(MIN_ALIASES, MIN).items(),
                        dict.fromkeys(MAX_ALIASES, MAX).items()))


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DataValidationError(ValueError):
    """Raised when some part of the multicriteria data (alternative matrix,
    criteria array or weights array) are not compatible with another part.

    """
    pass


# =============================================================================
# FUNCTIONS
# =============================================================================

def iter_equal(a, b):
    """Validate if two iterables are equals independently of their type."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.allclose(a, b, equal_nan=True)
    return a == b


def is_mtx(mtx, size=None):
    """Return True if mtx is two dimensional structure.

    If `size` is not None, must be a expected shape of the mtx

    """
    try:
        mtx = np.asarray(mtx)
        a, b = mtx.shape
        if size and (a, b) != size:
            return False
    except Exception:
        return False
    return True


def criteriarr(criteria):
    """Validate if the iterable only contains MIN (or any alias) and MAX
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

    Raises
    ------

    DataValidationError :
        if some value of the criteria array are not MIN (-1) or MAX (1)

    """

    pcriteria = np.array([ALIASES.get(c) for c in criteria])
    if None in pcriteria:
        msg = (
            "Criteria Array only accept minimize or maximize Values. Found {}")
        raise DataValidationError(msg.format(criteria))
    return pcriteria


def validate_data(mtx, criteria, weights=None):
    """Validate if the main components of the Data in scikit-criteria are
    compatible.

    The function tests:

    - The matrix (mtx) must be 2-dimensional.
    - The criteria array must be a criteria array (criteriarr function).
    - The number of criteria must be the same number of columns in mtx.
    - The weight array must be None or an iterable with the same length
      of the criteria.

    Parameters
    ----------

    mtx : 2D array-like
        2D alternative matrix, where every column (axis 0) are a criteria, and
        every row (axis 1) is an alternative.

    criteria : Array-like
        The sense of optimality of every criteria. Must has only
        MIN (-1) and MAX (1) values. Must has the same elements as columns
        has ``mtx``

    weights : array like or None
        The importance of every criteria. Must has the same elements as columns
        has ``mtx`` or None.

    Returns
    -------

    mtx : numpy.ndarray
        mtx representations as 2d numpy.ndarray.
    criteria : numpy.ndarray
        A criteria as numpy.ndarray.
    weights : numpy.ndarray or None
        A weights as numpy.ndarray or None (if weights is None).

    Raises
    ------

    DataValidationError :
        If the data are incompatible.

    """
    mtx = np.asarray(mtx)
    if not is_mtx(mtx):
        msg = "'mtx' must be a 2 dimensional array"
        raise DataValidationError(msg)

    criteria = criteriarr(criteria)
    if len(criteria) != np.shape(mtx)[1]:
        msg = "{} senses of optimality given but mtx has {} criteria".format(
            len(criteria), np.shape(mtx)[1])
        raise DataValidationError(msg)

    weights = (np.asarray(weights) if weights is not None else None)
    if weights is not None:
        if len(weights) != len(criteria):
            msg = "{} weights given for {} criteria".format(
                len(weights), len(criteria))
            raise DataValidationError(msg)

    return mtx, criteria, weights
