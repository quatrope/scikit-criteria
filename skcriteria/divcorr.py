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

"""Some wrapps around basic divergence and correlation functions to use with
alternative matrix

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from scipy import stats


# =============================================================================
# CONSTANTS
# =============================================================================

DIVERGENCE_FUNCTIONS = {}

CORRELATION_FUNCTIONS = {}

FUNCTIONS_TYPES = {
    "divergence": DIVERGENCE_FUNCTIONS,
    "correlation": CORRELATION_FUNCTIONS}


def register_stat(name, ftype):

    if ftype not in FUNCTIONS_TYPES:
        msg = "'ftype' must be one of {}. Found {}"
        raise ValueError(msg.format(FUNCTIONS_TYPES.keys(), ftype))

    def _dec(func):
        if not hasattr(func, "__call__"):
            raise TypeError("'func' must be callable")
        fdict = FUNCTIONS_TYPES[ftype]
        if name in fdict:
            msg = "{} function '{}' already exist"
            raise ValueError(msg.format(ftype, name))
        fdict[name] = func
        return func

    return _dec


# =============================================================================
# FUNCTIONS
# =============================================================================

@register_stat("std", "divergence")
def std(arr):
    return np.std(arr, axis=0)


@register_stat("var", "divergence")
def var(arr):
    return np.var(arr, axis=0)


@register_stat("pearson", "correlation")
def corr_pearson(arr):
    return np.corrcoef(arr)


@register_stat("spearman", "correlation")
def corr_spearman(arr):
    return stats.spearmanr(arr.T, axis=0).correlation
