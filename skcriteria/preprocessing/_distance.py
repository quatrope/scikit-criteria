#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for inverting minimization criteria and \
converting them into maximization ones.

In addition to the main functionality, an agnostic MCDA function is offered
that inverts columns of a matrix based on a mask.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ..base import SKCTransformerMixin
from ..data import Objective
from ..utils import doc_inherit

# =============================================================================
# FUNCTIONS
# =============================================================================


def cenit_distance(matrix, objectives):
    matrix = np.asarray(matrix, dtype=float)

    maxs = np.max(matrix, axis=0)
    mins = np.min(matrix, axis=0)

    where_max = np.equal(objectives, Objective.MAX.value)

    cenit = np.where(where_max, maxs, mins)
    nadir = np.where(where_max, mins, maxs)

    return (matrix - nadir) / (cenit - nadir)


class CenitDistance(SKCTransformerMixin):
    @doc_inherit(SKCTransformerMixin._transform_data)
    def _transform_data(self, matrix, objectives, **kwargs):

        distance_mtx = cenit_distance(matrix, objectives)

        dtypes = np.full(np.shape(objectives), float)

        kwargs.update(
            matrix=distance_mtx, objectives=objectives, dtypes=dtypes
        )
        return kwargs
