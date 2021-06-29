#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..data import DecisionMatrix, Objective

# =============================================================================
# FUNCTIONS
# =============================================================================


def minimize_to_maximize(mtx, objectives):

    new_mtx = np.array(mtx, dtype=float)

    minimize_mask = np.equal(objectives, Objective.MIN.value)

    inverted_values = 1.0 / new_mtx[:, minimize_mask]
    new_mtx[:, minimize_mask] = inverted_values

    new_objectives = (
        np.zeros(np.shape(objectives), dtype=int) + Objective.MAX.value
    )

    return new_mtx, new_objectives


# =============================================================================
# CLASS
# =============================================================================


class BaseDecisor:

    _decisor_type = None

    def validate_data(self, **kwargs):
        pass


class NormalizerMixin:

    _decisor_type = "normalizer"

    def normalize(self, dm):
        mtx = dm.mtx
        objectives = dm.objectives_values
        weights = dm.weights
        anames = dm.anames
        cnames = dm.cnames
        dtypes = dm.dtypes

        self.validate_data(
            mtx=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=dtypes,
        )

        nkwargs = self.transform(
            mtx=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=dtypes,
        )

        norm_dm = DecisionMatrix.from_mcda_data(**nkwargs)

        return norm_dm


class MinimizeToMaximize(NormalizerMixin, BaseDecisor):
    def transform(self, mtx, objectives, dtypes, **kwargs):
        inv_mtx, inv_objectives = minimize_to_maximize(mtx, objectives)

        # we are trying to preserve the original dtype as much as possible
        # only the minimize criteria are changed.
        inv_dtypes = np.where(
            objectives == Objective.MIN.value, inv_mtx.dtype, dtypes
        )

        kwargs.update(
            mtx=inv_mtx, objectives=inv_objectives, dtypes=inv_dtypes
        )
        return kwargs
