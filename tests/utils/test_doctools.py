#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.decorator"""


# =============================================================================
# IMPORTS
# =============================================================================

import string
import warnings

import numpy as np

import pytest

from skcriteria.utils import doctools

# =============================================================================
# TEST CLASSES
# =============================================================================


def test_doc_inherit():
    chars = tuple(string.ascii_letters + string.digits)
    random = np.random.default_rng(seed=42)

    doc = "".join(random.choice(chars, 1000))

    def func_a(): ...

    func_a.__doc__ = doc

    @doctools.doc_inherit(func_a)
    def func_b(): ...

    @doctools.doc_inherit(doc)
    def func_c(): ...

    assert doc == func_a.__doc__ == func_b.__doc__ == func_c.__doc__

    # test warnings
    with pytest.warns(UserWarning):

        @doctools.doc_inherit(doc, warn_class=True)
        class A:  # noqa
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @doctools.doc_inherit(doc, warn_class=False)
        class A:  # noqa
            pass
