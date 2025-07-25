#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.agg.similarity deprected module"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_similarity():
    with pytest.deprecated_call():
        from skcriteria.agg import similarity  # noqa

    from skcriteria.agg import topsis

    assert similarity.TOPSIS is topsis.TOPSIS
    assert similarity.topsis is topsis.topsis
