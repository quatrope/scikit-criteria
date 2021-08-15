#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for linnear programming based on PuLP.

This file contains an abstraction class to manipulate in a more OOP way
the underlining PuLP model

"""


# =============================================================================
# IMPORTS
# =============================================================================

import pulp

from .bunch import Bunch


# =============================================================================
# VARIABLES
# =============================================================================


class _Var(pulp.LpVariable):
    def __init__(self, name, low=None, up=None, *args, **kwargs):
        super(_Var, self).__init__(
            name=name,
            lowBound=low,
            upBound=up,
            cat=self.var_type,
            *args,
            **kwargs,
        )


class Float(_Var):
    var_type = pulp.LpContinuous


class Int(_Var):
    var_type = pulp.LpInteger


class Bool(_Var):
    var_type = pulp.LpBinary


# =============================================================================
# PROBLEM ABSTRACT CLASS
# =============================================================================


class _LPBase:
    def __init__(self, z, name="no-name", solver=None, **solver_kwds):
        problem = pulp.LpProblem(name, self.sense)
        if solver:
            if isinstance(solver, str):
                solver = pulp.getSolver(solver.upper(), **solver_kwds)
            problem.solver = solver

        problem += z, "Z"

        self._problem = problem

    def __repr__(self):
        """model.__repr__() <==> repr(model)."""
        cls_name = type(self).__name__
        objective = self._problem.objective
        constraints = ",\n  ".join(
            map(str, self._problem.constraints.values())
        )
        return f"{cls_name}({objective}).subject_to(\n  {constraints}\n)"

    @property
    def v(self):
        """Access to underlining variables."""
        return Bunch("variables", self._problem.variablesDict())

    def subject_to(self, *args):
        for c in args:
            self._problem += c
        return self

    def solve(self):
        problem = self._problem.copy()
        problem.solve()

        objective = pulp.value(problem.objective)

        variables, values = [], []
        for v in problem.variables():
            variables.append(v.name)
            values.append(v.varValue)

        status = pulp.LpStatus[problem.status]

        result_dict = {
            "lp_status_code": problem.status,
            "lp_status": status,
            "lp_objective": objective,
            "lp_variables": variables,
            "lp_values": values,
            "lp_problem": problem,
        }

        return Bunch(name="result", data=result_dict)


# =============================================================================
# CONCRETE CLASS
# =============================================================================


class Minimize(_LPBase):
    sense = pulp.LpMinimize


class Maximize(_LPBase):
    sense = pulp.LpMaximize
