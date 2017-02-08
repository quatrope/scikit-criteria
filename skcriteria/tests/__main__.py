#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# =============================================================================
# FUTURE
# =============================================================================

from __future__ import unicode_literals, absolute_import


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import argparse
import unittest
import importlib

from . import core, utils


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))


# =============================================================================
# FUNCTIONS
# =============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="Run the core test cases for scikit-crieteria")

    parser.add_argument(
        "-f", "--failfast", dest='failfast', default=False,
        help='Stop on first fail or error', action='store_true')

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-v", "--verbose", dest='verbosity',  default=0, const=1,
        help='Verbose output', action='store_const')
    group.add_argument(
        "-vv", "--vverbose", dest='verbosity', const=2,
        help='Verbose output', action='store_const')

    return parser


def load_test_modules():
    base_pkg, test_modules_names = ".".join(["skcriteria", "tests"]), []
    for dirpath, dirnames, filenames in os.walk(PATH):
        pkg = [] if dirpath == PATH else [os.path.basename(dirpath)]
        for fname in filenames:
            name, ext = os.path.splitext(fname)
            if name.startswith("test_") and ext == ".py":
                modname = ".".join(pkg + [name])
                test_modules_names.append((base_pkg, modname))

    test_modules = set()
    for pkg, modname in test_modules_names:
        dot_modname = ".{}".format(modname)
        module = importlib.import_module(dot_modname, pkg)
        test_modules.add(module)
    return tuple(test_modules)


def run_tests(verbosity=1, failfast=False):
    """Run test of scikit-criteria"""

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    runner = unittest.runner.TextTestRunner(
        verbosity=verbosity, failfast=failfast)

    load_test_modules()

    for testcase in utils.collect_subclasses(core.SKCriteriaTestCase):
        tests = loader.loadTestsFromTestCase(testcase)
        if tests.countTestCases():
                suite.addTests(tests)
    return runner.run(suite)


def main(argv):
    parser = create_parser()
    arguments = parser.parse_args(argv)

    # RUN THE TESTS
    result = run_tests(
        verbosity=arguments.verbosity, failfast=arguments.failfast)

    # EXIT WITH CORRECT STATUS
    sys.exit(not result.wasSuccessful())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main(sys.argv[1:])
