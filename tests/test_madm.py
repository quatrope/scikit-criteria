#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for the deprecated module skcriteria.madm

"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest


# =============================================================================
# TEST MADM
# =============================================================================


def test_deprecated_module_madm():
    with pytest.deprecated_call():
        from skcriteria import madm
    del madm
