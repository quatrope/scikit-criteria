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

import pulp

import attr


# =============================================================================
# CONSTANTS
# =============================================================================

SOLVERS = {
    "pulp": pulp.PULP_CBC_CMD,

    'coin': pulp.COIN,
    'coinmp_dll': pulp.COINMP_DLL,

    'cplex': pulp.CPLEX,
    # 'cplex_dll': pulp.CPLEX_DLL,
    'cplex_py': pulp.CPLEX_PY,

    'glpk': pulp.GLPK,
    'glpk_py': pulp.PYGLPK,

    'gurobi': pulp.GUROBI,
    'gurobi_cmd': pulp.GUROBI_CMD,

    'xpress': pulp.XPRESS,
    'yaposib': pulp.YAPOSIB
}


CMP = {
    pulp.LpMinimize: operator.ge,
    pulp.LpMaximize: operator.le
}


# =============================================================================
# RESULT CLASSES
# =============================================================================

@attr.s(frozen=True)
class Result(object):

    status_code = attr.ib(
        repr=False,
        validator=attr.validators.in_(pulp.constants.LpStatus.keys()))
    status = attr.ib(
        validator=attr.validators.in_(pulp.constants.LpStatus.values()))
    objective = attr.ib(
        validator=attr.validators.instance_of(float))
    variables = attr.ib(converter=tuple)
    values = attr.ib(converter=tuple)


# =============================================================================
# VARIABLES
# =============================================================================

class _Var(pulp.LpVariable):

    def __init__(self, name, low=None, up=None, *args, **kwargs):
        super(_Var, self).__init__(
            name=name, lowBound=low, upBound=up,
            cat=self.var_type, *args, **kwargs)


class Float(_Var):
    var_type = pulp.LpContinuous


class Int(_Var):
    var_type = pulp.LpInteger


class Bool(_Var):
    var_type = pulp.LpBinary


VAR_TYPE = {
    int: Int,
    float: Float,
    bool: Bool
}


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

        x_s = []
        for idx, xi in enumerate(x):
            vcls = VAR_TYPE.get(type(xi), Float)
            x = vcls("x{}".format(idx + 1), low=0)
            x_s.append(x)

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
        variables, values = [], []
        for v in self.variables():
            variables.append(v.name)
            values.append(v.varValue)
        status = pulp.LpStatus[self.status]
        self.assignVarsVals(dict.fromkeys(variables, None))
        return Result(status_code=self.status,
                      status=status,
                      objective=objective,
                      variables=variables,
                      values=values)


# =============================================================================
# CONCRETE CLASS
# =============================================================================

class Minimize(_LP):
    sense = pulp.LpMinimize


class Maximize(_LP):
    sense = pulp.LpMaximize
