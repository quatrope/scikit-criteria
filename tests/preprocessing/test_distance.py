#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.preprocessing.distance"""

# =============================================================================
# IMPORTS
# =============================================================================


import pytest

from skcriteria.preprocessing.distance import CenitDistance, cenit_distance


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_cenit_distance_deprecation_warning():
    with pytest.deprecated_call():
        cenit_distance([[1, 2, 3]], [max, max, max])


def test_CenitDistance_deprecation_warning():
    with pytest.deprecated_call():
        CenitDistance()
