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
# HELP
# =============================================================================

__doc__ = """Run the scikit-criteria tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import argparse

from .. import tests
from ..tests import core


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_parser():
    """Create a parser for tests from command line

    """
    global __doc__

    parser = argparse.ArgumentParser(prog=__file__, description=__doc__)
    parser.add_argument(
        "-f", "--failfast", dest="failfast", action="store_true",
        help="Fail on first test that fail"
    )
    parser.add_argument(
        "-m", "--modules", dest="modules", action="store",
        nargs="+", metavar="MODULE",
        help="List of modules to run (see -l/--listmodules option)"
    )
    parser.add_argument(
        "-v", "--verbosity", dest="verbosity", action="store", type=int,
        default=1, help="Verbosity level of test [0|1|2]"
    )
    parser.add_argument(
        "-l", "--listmodules", dest="listmodules", action="store_true",
        help="List available modules to run independently"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.listmodules:
        for modname in frozenset(testcls.modname() for testcls in
                                 core.SKCriteriaTestCase.subclasses()):
            print(modname)
    else:
        tests.run_tests(args.verbosity, args.modules, args.failfast)
