#!/usr/bin/env python
# -*- coding: utf-8 -*-

# License: 3 Clause BSD
# http://scikit-criteria.org/


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """All scikit-criteria base tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest
import random

import numpy as np
import numpy.testing as npt

import six
from six.moves import range


# =============================================================================
# BASE CLASS
# =============================================================================

class SKCriteriaTestCase(unittest.TestCase):

    def assertAllClose(self, a, b, **kwargs):
        return npt.assert_allclose(a, b, **kwargs)

    def assertArrayEqual(self, a, b, **kwargs):
        return npt.assert_array_equal(a, b, **kwargs)

    def assertAll(self, arr, **kwargs):
        assert np.all(arr), "'{}' is not all True".format(arr)

    def rrange(self, a, b):
        return range(random.randint(a, b))

    if six.PY2:
        assertRaisesRegex = six.assertRaisesRegex
        assertCountEqual = six.assertCountEqual
