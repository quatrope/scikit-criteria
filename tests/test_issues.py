#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tests for reported issues in the skcriteria issue tracker."""


# =============================================================================
# IMPORTS
# =============================================================================

import subprocess
import sys


# =============================================================================
# TESTS
# =============================================================================


def test_issue76():
    """Test for issue #76: skcriteria should not delete importlib.metadata.

    Ensures that importing skcriteria does not remove `importlib.metadata`
    from the global namespace. Previously, skcriteria was deleting
    `importlib.metadata` during initialization, which broke other packages
    that depended on it (e.g., when calling `importlib.metadata.version()`).

    This test runs in a separate Python subprocess to ensure a clean
    environment where `importlib.metadata` has not been previously imported
    by the test framework itself.

    See: https://github.com/quatrope/scikit-criteria/issues/76
    """
    commands = ";".join(
        [
            "import importlib.metadata",
            "metadata = importlib.metadata",
            "import skcriteria as skc",
            "assert importlib.metadata is metadata",
            "assert 'metadata' not in dir(skc)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", commands],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
