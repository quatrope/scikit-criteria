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

"""Utilities for linnear programming based on PuLP

"""


# =============================================================================
# IMPORTS
# =============================================================================

import operator
from collections import Mapping

import pulp

import attr


# =============================================================================
# CONSTANTS
# =============================================================================

VAR_TYPE = {
    int: pulp.LpInteger,
    float: pulp.LpContinuous,
    bool: pulp.LpBinary
}

SOLVERS = {
    "pulp": pulp.solvers.PULP_CBC_CMD,

    'coin': pulp.solvers.COIN,
    'coinmp_dll': pulp.solvers.COINMP_DLL,

    'cplex': pulp.solvers.CPLEX,
    'cplex_dll': pulp.solvers.CPLEX_DLL,
    'cplex_py': pulp.solvers.CPLEX_PY,

    'glpk': pulp.solvers.GLPK,
    'glpk_py': pulp.solvers.PYGLPK,

    'gurobi': pulp.solvers.GUROBI,
    'gurobi_cmd': pulp.solvers.GUROBI_CMD,

    'scip': pulp.solvers.SCIP,
    'xpress': pulp.solvers.XPRESS,
    'yaposib': pulp.solvers.YAPOSIB
}


CMP = {
    pulp.LpMinimize: operator.ge,
    pulp.LpMaximize: operator.le
}


# =============================================================================
# RESULT CLASSES
# =============================================================================

class Bunch(dict):

    def __getattr__(self, aname):
        try:
            return self[aname]
        except KeyError as err:
            raise AttributeError(*err.args)


@attr.s(frozen=True)
class Result(object):

    status_code = attr.ib(
        repr=False,
        validator=attr.validators.in_(pulp.constants.LpStatus.keys()))
    status = attr.ib(
        validator=attr.validators.in_(pulp.constants.LpStatus.values()))
    objective = attr.ib(
        validator=attr.validators.instance_of(float))
    variables = attr.ib(converter=Bunch)


class Var(pulp.LpVariable):

    def __init__(self, name, low=None, up=None, type=float, *args, **kwargs):
        super(Var, self).__init__(
            name=name, lowBound=low, upBound=up,
            cat=VAR_TYPE[type], *args, **kwargs)


# =============================================================================
# PROBLEM ABSTRACT CLASS
# =============================================================================

class _LP(pulp.LpProblem):

    def __init__(self, z, name="no-name", solver=None, **solver_kwds):
        super(_LP, self).__init__(name, self.sense)
        if solver:
            if isinstance(solver, str):
                cls = SOLVERS[solver]
                solver = cls(**solver_kwds) if cls else None
            self.solver = solver
        self += z, "Z"

    @classmethod
    def frommtx(cls, c, A, b, x=None, *args, **kwargs):
        # variables
        if x is None:
            x_n = max([len(c), len(b)] + [len(e) for e in A])
            x = [None] * x_n
        else:
            x_n = len(x)

        xkwds = []
        for xi in x:
            if xi is None:
                xi = {"low": 0, "type": float}
            elif xi in VAR_TYPE:
                xi = {"low": 0, "type": xi}
            xkwds.append(xi)

        x_s = [Var("x{}".format(idx + 1), **xkwds[idx]) for idx in range(x_n)]

        # Z
        z = sum([ci * xi for ci, xi in zip(c, x_s)])

        # restrictions:
        cmp = CMP[cls.sense]

        restrictions = []
        for row, bi in zip(A, b):

            left = sum([ri * xi for ri, xi in zip(row, x_s)])
            restrictions.append(cmp(left, bi))

        model = cls(z=z, *args, **kwargs)
        model._buff = x_s

        return model.sa(*restrictions)

    def sa(self, *args, **kwargs):
        for c in args:
            self += c
        for n, c in kwargs.items():
            self += c, n
        return self

    def solve(self):
        super(_LP, self).solve()
        objective = pulp.value(self.objective)
        variables = {v.name: v.varValue for v in self.variables()}
        status=pulp.LpStatus[self.status]
        model.assignVarsVals(dict.fromkeys(variables, None))
        return Result(status_code=self.status,
                      status=status,
                      objective=objective,
                      variables=variables)


# =============================================================================
# CONCRETE CLASS
# =============================================================================

class Minimize(_LP):
    sense = pulp.LpMinimize


class Maximize(_LP):
    sense = pulp.LpMaximize
