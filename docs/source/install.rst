.. _installation-instructions:

==========================
Installation
==========================

Using conda
-----------

The easiest and fastest way to get the package up and running is to
install scikit-criteria using `conda`_:

.. code:: bash

   $ conda install -c conda-forge scikit-criteria

or, better yet, using `mamba`_, which is a super fast replacement for
``conda``:

.. code:: bash

   $ conda install -c conda-forge mamba
   $ mamba install -c conda-forge scikit-criteria

.. note::

   We encourage users to use conda or mamba and the
   `conda-forge <https://conda-forge.org/>`_ packages for convenience,
   especially when developing on Windows. It is recommended to create a new
   environment.

If the installation fails for any reason, please open an issue in the
`issue tracker`_.

Alternative installation methods
--------------------------------

You can also `install scikit-criteria from PyPI`_ using pip:

.. code:: bash

   $ pip install scikit-criteria

Finally, you can also install the latest development version of
scikit-criteria `directly from GitHub`_:

.. code:: bash

   $ pip install git+https://github.com/quatrope/scikit-criteria/

This is useful if there is some feature that you want to try, but we did
not release it yet as a stable version. Although you might find some
unpolished details, these development installations should work without
problems. If you find any, please open an issue in the `issue tracker`_.

.. warning::

   It is recommended that you
   **never ever use sudo** with distutils, pip, setuptools and friends in Linux
   because you might seriously break your system
   [`1 <http://wiki.python.org/moin/CheeseShopTutorial#Distutils_Installation>`_]
   [`2 <http://stackoverflow.com/questions/4314376/how-can-i-install-a-python-egg-file/4314446#comment4690673_4314446>`_]
   [`3 <http://workaround.org/easy-install-debian>`_]
   [`4 <http://matplotlib.1069221.n5.nabble.com/Why-is-pip-not-mentioned-in-the-Installation-Documentation-tp39779p39812.html)>`_].
   Use `virtual environments <https://docs.python.org/3/library/venv.html>`_ instead.

.. _conda: https://conda.io/docs/
.. _mamba: https://mamba.readthedocs.io/
.. _issue tracker: https://github.com/quatrope/scikit-criteria/issues
.. _install scikit-criteria from PyPI: https://pypi.python.org/pypi/scikit-criteria/
.. _directly from GitHub: https://github.com/quatrope/scikit-criteria/


If you don't have Python
-------------------------

If you don't already have a python installation with numpy and scipy, we
recommend to install either via your package manager or via a python bundle.
These come with numpy, scipy, matplotlib and many other helpful
scientific and data processing libraries.

`Canopy
<https://www.enthought.com/products/canopy>`_ and `Anaconda
<https://www.continuum.io/downloads>`_ both ship a recent
version of Python, in addition to a large set of scientific python
library for Windows, Mac OSX and Linux.
