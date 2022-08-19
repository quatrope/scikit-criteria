#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Multiple context managers to use inside scikit-criteria."""

# =============================================================================
# IMPORTS
# =============================================================================

import contextlib

# =============================================================================
# FUNCTIONS
# =============================================================================


@contextlib.contextmanager
def df_temporal_header(df, header, name=None):
    """Temporarily replaces a DataFrame columns names.

    Optionally also assign another name to the columns.

    Parameters
    ----------
    header : sequence
        The new names of the columns.
    name : str or None (default None)
        New name for the index containing the columns in the DataFrame. If
        'None' the original name of the columns present in the DataFrame is
        preserved.

    """
    original_header = df.columns
    original_name = original_header.name

    name = original_name if name is None else name
    try:
        df.columns = header
        df.columns.name = name
        yield df
    finally:
        df.columns = original_header
        df.columns.name = original_name
