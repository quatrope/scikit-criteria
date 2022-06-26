#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Normalization through the distance to distance function."""


# =============================================================================
# IMPORTS
# =============================================================================


import functools

from . import scalers
from ..utils import deprecated, doc_inherit

# =============================================================================
# OLD SCALERS
# =============================================================================


@deprecated(
    reason="Use `skcriteria.preprocessing.scalers.cenit_distance` instead,",
    version=0.8,
)
@doc_inherit(scalers.cenit_distance)
@functools.wraps(scalers.cenit_distance)
def cenit_distance(*args, **kwargs):
    return scalers.cenit_distance(*args, **kwargs)


@deprecated(
    reason="Use `skcriteria.preprocessing.scalers.CenitDistance` instead,",
    version=0.8,
)
@doc_inherit(scalers.CenitDistance)
class CenitDistance(scalers.CenitDistance):
    pass
