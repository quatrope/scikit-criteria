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

import numpy as np

import pulp

from .bunch import Bunch
from .decorators import doc_inherit


# =============================================================================
# UTILITIES
# =============================================================================


def is_solver_available(solver):
    """Return True if the solver is available."""
    return solver is None or solver.upper() in ["PULP"] + pulp.list_solvers(
        onlyAvailable=True
    )


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
    """:class:`pulp.LpVariable` with :class:`pulp.LpContinuous` category.

    Example
    -------
    This two codes are equivalent.

    .. code-block:: python

        x = pulp.LpVariable("x", cat=pulp.LpContinuous)  # pure PuLP
        x = lp.Float("x")  # skcriteria.utils.lp version

    """

    var_type = pulp.LpContinuous


class Int(_Var):
    """:class:`pulp.LpVariable` with :class:`pulp.LpInteger` category.

    Example
    -------
    This two codes are equivalent.

    .. code-block:: python

        x = pulp.LpVariable("x", cat=pulp.LpInteger)  # pure PuLP
        x = lp.Int("x")  # skcriteria.utils.lp version

    """

    var_type = pulp.LpInteger


class Bool(_Var):
    """:class:`pulp.LpVariable` with :class:`pulp.LpBinary` category.

    Example
    -------
    This two codes are equivalent.

    .. code-block:: python

        x = pulp.LpVariable("x", cat=pulp.LpBinary)  # pure PuLP
        x = lp.Bool("x")  # skcriteria.utils.lp version

    """

    var_type = pulp.LpBinary


# =============================================================================
# PROBLEM ABSTRACT CLASS
# =============================================================================


class _LPBase:
    """Creates a LP problem with a way better sintax than PuLP.

    Parameters
    ----------
    z: :class:`LpAffineExpression`
        A linear combination of :class:`LpVariables<LpVariable>`.
    name: str (default="no-name")
        Name of the problem.
    solver: None, str or any :class:`pulp.LpSolver` instance (default=None)
        Solver of the problem. If it's None, the default solver is used.
        PULP is an alias os None.
    solver_kwds: dict
        Dictionary of keyword arguments for the solver.

    Example
    -------

    .. code-block:: python

        # variable declaration
        x0 = lp.Float("x0", low=0)
        x1 = lp.Float("x1", low=0)
        x2 = lp.Float("x2", low=0)

        # model
        model = lp.Maximize(  # or lp.Minimize
            z=250 * x0 + 130 * x1 + 350 * x2
        )

        # constraints
        model.subject_to(
            120 * x0 + 200 * x1 + 340 * x2 <= 500,
            -20 * x0 + -40 * x1 + -15 * x2 <= -15,
            800 * x0 + 1000 * x1 + 600 * x2 <= 1000,
        )

    Also you can create the model and the constraints in one "line".

    .. code-block:: python

        model = lp.Maximize(   or lp.Minimize
            z=250 * x0 + 130 * x1 + 350 * x2, solver=solver
        ).subject_to(
            120 * x0 + 200 * x1 + 340 * x2 <= 500,
            -20 * x0 + -40 * x1 + -15 * x2 <= -15,
            800 * x0 + 1000 * x1 + 600 * x2 <= 1000,
        )

    """

    def __init__(self, z, name="no-name", solver=None, **solver_kwds):
        """Create an instance of problem solver."""
        problem = pulp.LpProblem(name, self.sense)
        if solver is not None and solver.upper() != "PULP":  # None == PULP
            if isinstance(solver, str):
                solver = pulp.getSolver(solver.upper(), **solver_kwds)
            problem.setSolver(solver)

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
        """Add a constraint to a underliying puLP problem.

        Parameters
        ----------
        args: tuple
            Multiple :class:`LpAffineExpression`

        Returns
        -------
        self:
            Return the same instance.


        """
        for c in args:
            self._problem += c
        return self

    def solve(self):
        """Solve the underlying problem and create a report as a dict.

        The method copy the original problem, and then solve it.

        Returns
        -------
        result: dict-like.
            Report of the problem as dict-like.

        """
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
            "lp_variables": np.asarray(variables),
            "lp_values": np.asarray(values),
            "lp_problem": problem,
        }

        return Bunch(name="result", data=result_dict)


# =============================================================================
# CONCRETE CLASS
# =============================================================================


@doc_inherit(_LPBase)
class Minimize(_LPBase):
    """Creates a Minimize LP problem with a way better sintax than PuLP."""

    sense = pulp.LpMinimize


@doc_inherit(_LPBase)
class Maximize(_LPBase):
    """Creates a Maximize LP problem with a way better sintax than PuLP."""

    sense = pulp.LpMaximize
