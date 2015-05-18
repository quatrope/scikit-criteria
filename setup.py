#!/usr/bin/env python
# -*- coding: utf-8 -*-

# "THE WISKEY-WARE LICENSE":
# <jbc.develop@gmail.com> and <nluczywo@gmail.com>
# wrote this file. As long as you retain this notice you can do whatever you
# want with this stuff. If we meet some day, and you think this stuff is worth
# it, you can buy me a WISKEY in return Juan BC and Nadia AL.


#==============================================================================
# DOCS
#==============================================================================

"""This file is for distribute scikit-criteria with distutils

"""


#==============================================================================
# FUNCTIONS
#==============================================================================

if __name__ == "__main__":
    import os
    import sys

    from ez_setup import use_setuptools
    use_setuptools()

    from setuptools import setup, find_packages

    setup(
        name="scikit-criteria",
        version="0.0.1",
        description="Multiple-criteria decision analysis package",
        author="JuanBC - NadiaAL",
        author_email="jbc.develop@gmail.com",
        url="http://scikit-criteria.org/",
        license="WISKEY-WARE",
        keywords="mcda mcdm ahp moora muti criteria".split(),
        classifiers=[],
        packages=[
            pkg for pkg in find_packages() if pkg.startswith("skcriteria")],
        include_package_data=True,
        py_modules=["ez_setup"],
        install_requires=["numpy", "scipy"],
    )
