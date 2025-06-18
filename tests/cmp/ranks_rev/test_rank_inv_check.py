#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for the deprecated module skcriteria.cmp.ranks_rev.rank_inv_check"""


# =============================================================================
# IMPORTS
# =============================================================================

from skcriteria.cmp import ranks_rev


# =============================================================================
# TEST MADM
# =============================================================================


def test_deprecated_ranks_rev():
    pattern = (
        ".. deprecated:: 0.9\n"
        "    'skcriteria.cmp.ranks_rev' package is deprecated, use "
        "'skcriteria.ranksrev' instead\n"
    )
    assert ranks_rev.__doc__.endswith(pattern)
