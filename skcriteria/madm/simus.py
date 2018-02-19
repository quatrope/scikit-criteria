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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """Simus methods"""

__all__ = [
    "SIMUS"]


# =============================================================================
# IMPORTS
# =============================================================================

import operator

import numpy as np

import joblib

from ..validate import MAX, MIN
from ..utils import lp
from ..utils.doc_inherit import doc_inherit

from ._dmaker import DecisionMaker


# =============================================================================
# FUNCTIONS
# =============================================================================

def make_stage(mtx, b, senses, zindex, solver):
    # retrieve the problem class
    problem = lp.Minimize if senses[zindex] == MIN else lp.Maximize

    # create the variables
    xs = tuple(
        lp.Float("x{}".format(idx), low=0) for idx in range(mtx.shape[1]))

    # create the objective function
    z_coef = mtx[zindex]
    z = sum(c * x for c, x in zip(z_coef, xs))

    # the conditions
    conditions = []
    for idx in range(mtx.shape[0]):
        if idx == zindex:
            continue
        coef = mtx[idx]

        left = sum(c * x for c, x in zip(coef, xs))
        op = operator.le if senses[idx] == MAX else operator.ge
        right = b[idx]

        condition = op(left, right)
        conditions.append(condition)
    stage = problem(z=z, solver=solver).sa(*conditions)
    return stage


def solve(stage):
    return stage.solve()


def simus(nmtx, ncriteria, nweights, b=None, solver="pulp", njobs=None):

    t_nmtx = nmtx.T

    b = np.asarray(b)
    if None in b:
        mins = np.min(t_nmtx, axis=1)
        maxs = np.max(t_nmtx, axis=1)

        auto_b = np.where(ncriteria == MAX, maxs, mins)
        b = np.where(b.astype(bool), b, auto_b)

    njobs = njobs or joblib.cpu_count()
    with joblib.Parallel(n_jobs=njobs) as jobs:
        stages = jobs(
            joblib.delayed(make_stage)(
                mtx=t_nmtx, b=b, senses=ncriteria, zindex=idx, solver=solver)
            for idx in range(t_nmtx.shape[0]))

        results = jobs(
            joblib.delayed(solve)(stage) for stage in stages)

        return None, results


# =============================================================================
# OO
# =============================================================================

class SIMUS(DecisionMaker):
    r"""

    """

    def __init__(self, mnorm="none", wnorm="none", solver="pulp", njobs=None):
        super(SIMUS, self).__init__(mnorm=mnorm, wnorm=wnorm)
        self._solver = solver
        self._njobs = njobs

    @doc_inherit
    def solve(self, ndata, b):
        nmtx, ncriteria, nweights = ndata.mtx, ndata.criteria, ndata.weights
        rank, points = simus(
            nmtx, ncriteria, nweights,
            b=b, solver=self._solver, njobs=self._njobs)
        return None, rank, {"points": points}

    @property
    def solver(self):
        return self._solver

    @property
    def njobs(self):
        return self._njobs


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
