#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Normalization through the distance to distance function.

This entire module is deprecated.

"""


# =============================================================================
# IMPORTS
# =============================================================================

from . import scalers
from ..utils import deprecated, doc_inherit


# =============================================================================
# CENIT DISTANCE
# =============================================================================

_skc_prep_scalers = "skcriteria.preprocessing.scalers"


@deprecated(
    reason=(
        f"Use ``{_skc_prep_scalers}.matrix_scale_by_cenit_distance`` instead"
    ),
    version=0.8,
)
@doc_inherit(scalers.matrix_scale_by_cenit_distance)
def cenit_distance(matrix, objectives):
    return scalers.matrix_scale_by_cenit_distance(matrix, objectives)


@deprecated(
    reason=f"Use ``{_skc_prep_scalers}.CenitDistanceMatrixScaler`` instead",
    version=0.8,
)
@doc_inherit(scalers.CenitDistanceMatrixScaler, warn_class=False)
class CenitDistance(scalers.CenitDistanceMatrixScaler):
    ...
