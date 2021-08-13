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
    "numpy", "scipy", "mock", "tabulate", "matplotlib", "pulp",
    "json-tricks", "joblib", "attrs", "pytest"
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
