#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""MCDA aggregation methods and internal machinery.

This Deprecated backward compatibility layer around skcriteria.agg.

"""

# =============================================================================
# IMPORT AND PATCH
# =============================================================================

# import the real agg package
from . import agg, utils

# this will be used in two places
__deprecation_conf = {
    "reason": (
        "'skcriteria.madm' module is deprecated, "
        "use 'skcriteria.agg' instead"
    ),
    "version": "0.8.5",
}


utils.deprecate.warn(**__deprecation_conf)

# store the metadata to preserve
__preserve = {
    "__name__": __name__,
    "__doc__": utils.deprecate.add_sphinx_deprecated_directive(
        __doc__, **__deprecation_conf
    ),
}


# update the globals
globals().update(agg.__dict__)
globals().update(__preserve)


# delete the unused modules and variables
del agg, __preserve, __deprecation_conf
