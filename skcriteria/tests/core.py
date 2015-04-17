#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> and <nluczywo@gmail.com>
# wrote this file. As long as you retain this notice you can do whatever you
# want with this stuff. If we meet some day, and you think this stuff is worth
# it, you can buy me a WISKEY in return Juan BC and Nadia AL.


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

"""All skcriteria test base"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest
import random

import numpy as np


# =============================================================================
# BASE CLASS
# =============================================================================

class SKCriteriaTestCase(unittest.TestCase):

    @classmethod
    def subclasses(cls):

        def collect(basecls):
            collected = set()
            for subcls in basecls.__subclasses__():
                collected.add(subcls)
                collected.update(collect(subcls))
            return collected

        return tuple(collect(cls))

    @classmethod
    def modname(cls):
        return cls.__module__.rsplit(".", 1)[-1].replace("test_", "", 1)

    # =========================================================================
    # UTILS
    # =========================================================================

    def rrange(self, li, ls):
        top = random.randint(li, ls)
        return xrange(top)

    # =========================================================================
    # ASSERTS
    # =========================================================================

    def assertIsClose(self, a, b, **kwargs):
        if not np.all(np.isclose(a, b, **kwargs)):
            msg = "'{}' != '{}'".format(a, b)
            raise AssertionError(msg)

    def assertAllClose(self, a, b, **kwargs):
        if not np.allclose(a, b, **kwargs):
            msg = "'{}' != '{}'".format(a, b)
            raise AssertionError(msg)

    def assertAll(self, arr, **kwargs):
        if not np.all(arr):
            msg = "'{}' is not all True".format(arr)
            raise AssertionError(msg)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
