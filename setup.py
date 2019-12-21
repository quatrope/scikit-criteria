#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2019, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute scikit-criteria

"""


# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

os.environ["__SKCRITERIA_IN_SETUP__"] = "True"
import skcriteria


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy", "scipy", "attrs", "tabulate"
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name=skcriteria.NAME,
        version=skcriteria.VERSION,
        description=skcriteria.DOC,
        author=skcriteria.AUTHORS,
        author_email=skcriteria.EMAIL,
        url=skcriteria.URL,
        license=skcriteria.LICENSE,
        keywords=skcriteria.KEYWORDS,
        classifiers=(
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ),
        packages=[
            pkg for pkg in find_packages() if pkg.startswith("skcriteria")],
        py_modules=["ez_setup"],
        install_requires=REQUIREMENTS,
    )


def do_publish():
    pass


if __name__ == "__main__":
    if sys.argv[-1] == 'publish':
        do_publish()
    else:
        do_setup()
