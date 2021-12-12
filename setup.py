#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute scikit-criteria

"""


# =============================================================================
# IMPORTS
# =============================================================================


import os
import pathlib

from setuptools import find_packages, setup

os.environ["__SKCRITERIA_IN_SETUP__"] = "True"
import skcriteria  # noqa

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy",
    "pandas",
    "pyquery",
    "scipy",
    "jinja2",
    "custom_inherit",
    "seaborn",
    "pulp",
]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name="scikit-criteria",
        version=skcriteria.VERSION,
        description=skcriteria.DOC,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="Juan B Cabral, Nadial Luczywo and QuatroPe",
        author_email="jbcabral@unc.edu.ar",
        url="http://scikit-criteria.org/",
        license="3 Clause BSD",
        keywords=[
            "muticriteria",
            "mcda",
            "mcdm",
            "weightedsum",
            "weightedproduct",
            "simus",
            "topsis",
            "moora",
            "electre",
            "critic",
            "entropy",
            "dominance",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ],
        packages=[
            pkg for pkg in find_packages() if pkg.startswith("skcriteria")
        ],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":
    do_setup()
