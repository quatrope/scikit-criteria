.. _installation-instructions:

==========================
Installation
==========================

The easiest way to install scikit-criteria is using ``pip`` ::

    pip install -U scikit-criteria


If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).
Building numpy and scipy from source can be complex (especially on Windows) and
requires careful configuration to ensure that they link against an optimized
implementation of linear algebra routines.
Instead, use a third-party distribution as described below.


Third-party Distributions
==========================
If you don't already have a python installation with numpy and scipy, we
recommend to install either via your package manager or via a python bundle.
These come with numpy, scipy, matplotlib and many other helpful
scientific and data processing libraries.

Available options are:

Canopy and Anaconda for all supported platforms
-----------------------------------------------

`Canopy
<https://www.enthought.com/products/canopy>`_ and `Anaconda
<https://www.continuum.io/downloads>`_ both ship a recent
version of Python, in addition to a large set of scientific python
library for Windows, Mac OSX and Linux.
