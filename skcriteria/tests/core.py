#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# DOC
# =============================================================================

"""All scikit-criteria base tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest
import random

import numpy.testing as npt

import six
from six.moves import range


# =============================================================================
# BASE CLASS
# =============================================================================

class SKCriteriaTestCase(unittest.TestCase):

    def assertIsClose(self, a, b, **kwargs):
        return npt.assert_allclose(a, b, **kwargs)

    def assertAllClose(self, a, b, **kwargs):
        return npt.assert_allclose(a, b, **kwargs)

    def assertAll(self, arr, **kwargs):
        assert np.all(arr), "'{}' is not all True".format(arr)

    def rrange(self, a, b):
        return range(random.randint(a, b))

    if six.PY2:
        assertRaisesRegex = six.assertRaisesRegex
        assertCountEqual = six.assertCountEqual

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
