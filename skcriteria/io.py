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
# IMPORTS
# =============================================================================

import sys
import datetime as dt
import platform

import numpy as np

import json_tricks as jt

from . import dmaker, VERSION

# =============================================================================
# CONTANTS
# =============================================================================

SKCF_VERSION = 1

CT_DECISION = "decision"

CT_DMAKER = "decision-maker"

DEFAULT_ALLOW_NAN = True

DEFAULT_INDENT = None

DEFAULT_PRESERVE_ORDER = False


# =============================================================================
# FUNCTIONS
# =============================================================================

def dumpd(obj):
    data = {}

    # check the content type
    if isinstance(obj, dmaker.Decision):
        data["content-type"] = CT_DECISION
    elif isinstance(obj, dmaker.DecisionMaker):
        data["content-type"] = CT_DMAKER
    else:
        msg = ("'obj' must be an istance of Decision or DecisionMaker. "
               "Found: {}").format(type(obj))
        raise TypeError(msg)

    # add the common data
    data.update({
        "data": obj,
        "version": SKCF_VERSION,
        "env": {
            "skcriteria": VERSION,
            "python": sys.version,
            "numpy": np.version.full_version,
            "platform": platform.platform()
        },
        "utc_created_at": dt.datetime.utcnow().isoformat()})
    return data


def dumps(obj, **kwargs):
    kwargs.setdefault("allow_nan", DEFAULT_ALLOW_NAN)
    kwargs.setdefault("indent", DEFAULT_INDENT)

    data = dumpd(obj)
    return jt.dumps(data, **kwargs)


def dump(obj, fp, **kwargs):
    kwargs.setdefault("allow_nan", DEFAULT_ALLOW_NAN)
    kwargs.setdefault("indent", DEFAULT_INDENT)

    data = dumpd(obj)
    return jt.dump(data, fp, **kwargs)


def loads(string, skcm_metadata=False, **kwargs):
    kwargs.setdefault("preserve_order", DEFAULT_PRESERVE_ORDER)

    data = jt.loads(string, **kwargs)
    return data if skcm_metadata else data["data"]


def load(fp, skcm_metadata=False, **kwargs):
    kwargs.setdefault("preserve_order", DEFAULT_PRESERVE_ORDER)

    data = jt.load(fp, **kwargs)
    return data if skcm_metadata else data["data"]
