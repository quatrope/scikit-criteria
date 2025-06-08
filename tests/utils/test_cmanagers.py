#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.utils.cmanagers"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

import pandas as pd

import pytest

from skcriteria.utils import cmanagers


# =============================================================================
# TEST CLASSES
# =============================================================================


def test_df_temporal_header():
    df = pd.DataFrame({"x": [1], "y": [2]})
    df.columns.name = "original"

    with cmanagers.df_temporal_header(df, ["a", "b"], "replaced") as df:
        pd.testing.assert_index_equal(
            df.columns, pd.Index(["a", "b"], name="replaced")
        )

    pd.testing.assert_index_equal(
        df.columns, pd.Index(["x", "y"], name="original")
    )


def test_hidden():
    code = """
from  skcriteria.utils import hidden
with hidden():
    import os
    import pandas as pd

os = 1
"""

    co_obj = compile(code, "test.py", "exec")
    ns = {}

    eval(co_obj, ns, ns)

    assert ns["os"] == 1
    assert ns["pd"] is pd

    assert ns["__dir__"].hidden_objects == {
        "os": os,
        "pd": pd,
        "hidden": cmanagers.hidden,
    }

    assert "pd" not in ns["__dir__"]()
    assert "hidden" not in ns["__dir__"]()
    assert "os" in ns["__dir__"]()


def test_hidden_hide_this_False():
    code = """
from  skcriteria.utils import hidden
with hidden(hide_this=False):
    import os
    import pandas as pd


"""

    co_obj = compile(code, "test.py", "exec")
    ns = {}

    eval(co_obj, ns, ns)

    assert ns["os"] == os
    assert ns["pd"] is pd

    assert ns["__dir__"].hidden_objects == {"os": os, "pd": pd}

    assert "pd" not in ns["__dir__"]()
    assert "os" not in ns["__dir__"]()
    assert "hidden" in ns["__dir__"]()


def test_hidden_dry_True():
    code = """
from  skcriteria.utils import hidden
with hidden(dry=True):
    import os
    import pandas as pd


"""

    co_obj = compile(code, "test.py", "exec")
    ns = {}

    eval(co_obj, ns, ns)

    assert ns["os"] == os
    assert ns["pd"] is pd
    assert ns["hidden"] is cmanagers.hidden

    assert "__dir__" not in ns


def test_hidden_two_times_fails():
    code = """
from  skcriteria.utils import hidden
with hidden():
    import os
    import pandas as pd

with hidden():
    import os
    import pandas as pd
"""

    co_obj = compile(code, "test.py", "exec")

    with pytest.raises(cmanagers.HiddenAlreadyUsedInThisContext):
        eval(co_obj)


def test_hidden_no_global_fails():
    code = """
class A:
    from  skcriteria.utils import hidden
    with hidden():
        import os
        import pandas as pd

"""
    co_obj = compile(code, "test.py", "exec")

    with pytest.raises(cmanagers.NonGlobalHidden):
        eval(co_obj)
