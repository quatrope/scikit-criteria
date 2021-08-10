#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.bunch

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from skcriteria.utils import rank


# =============================================================================
# TEST Bunch
# =============================================================================


def test_rank():
    values = [0.5, 0.2, 0.6, 0.8]
    expected = [2, 1, 3, 4]
    result = rank(values)
    assert np.all(result == expected)


def test_rank_reverse():
    values = [0.5, 0.2, 0.6, 0.8]
    expected = [3, 4, 2, 1]
    result = rank(values, reverse=True)
    assert np.all(result == expected)
