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

__doc__ = """All scikit-criteria tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import unittest

from . import (
    core,
    test_common,
    test_wsum,
    test_wprod,
    test_moora,
    test_ahp
)


# =============================================================================
# FUNCTIONS
# =============================================================================

def run_tests(verbosity=1, modules=None, failfast=False):
    """Run test of scikit-criteria

    """
    def collect_modules():
        modules = {}
        for testcls in core.SKCriteriaTestCase.subclasses():
            modname = testcls.modname()
            if modname not in modules:
                modules[modname] = set()
            modules[modname].add(testcls)
        return modules

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    runner = unittest.runner.TextTestRunner(
        verbosity=verbosity, failfast=failfast
    )
    for modname, testcases in collect_modules().items():
        if not modules or modname in modules:
            for testcase in testcases:
                tests = loader.loadTestsFromTestCase(testcase)
                if tests.countTestCases():
                        suite.addTests(tests)
    return runner.run(suite)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
