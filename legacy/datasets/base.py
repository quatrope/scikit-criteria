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

"""Base IO code for all datasets

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

import numpy as np

from ..validate import MIN, MAX
from ..base import Data


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(PATH, "data")


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_camera():
    """A dataset of about 1000 cameras with 13 properties such as
    weight, focal length, price, etc.

    Notes
    -----

    These dataset have been gathered and cleaned up by Petra
    Isenberg, Pierre Dragicevic and Yvonne Jansen.
    https://perso.telecom-paristech.fr/eagan/class/igr204/datasets

    """
    path = os.path.join(DATA_PATH, "Camera.csv")
    dtypes = [
        ("Model", str, 35),
        ("Release date", float),
        ("Max resolution", float),
        ("Low resolution", float),
        ("Effective pixels", float),
        ("Zoom wide (W)", float),
        ("Zoom tele (T)", float),
        ("Normal focus range", float),
        ("Macro focus range", float),
        ("Storage included", float),
        ("Weight (inc. batteries)", float),
        ("Dimensions", float),
        ("Price", float),
    ]

    criteria = [MAX, MAX, MAX, MAX, MAX, MAX, MAX, MAX, MAX, MIN, MIN, MIN]

    data = np.genfromtxt(path, delimiter=";", dtype=dtypes, skip_header=2)

    # columns of the alternative matrix
    columns = list(data.dtype.names[1:])

    anames = data["Model"]
    cnames = [dt[0] for dt in dtypes[1:]]

    mtx = np.asarray(data[columns].tolist())

    meta = {"desc": load_camera.__doc__}

    data = Data(mtx=mtx, criteria=criteria,
                anames=anames, cnames=cnames, meta=meta)
    return data
