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

"""Core functionalities of all madm methods.

"""


# =============================================================================
# IMPORTS
# =============================================================================

import operator
import uuid
from collections.abc import Mapping

import numpy as np

from tabulate import tabulate

from ..validate import iter_equal
from ..base import TABULATE_PARAMS, BaseSolver


# =============================================================================
# EXTRA
# =============================================================================

class Extra(Mapping):
    """The Extra object is a dictionary with steroids. The main objective
    is to store and present the extra data created by different madm methods.

    Parameters
    ----------
    data : any object supported for create a dict.
        - Extra(dict) -> new Extra initialized from a dict object.
        - Extra(mapping) -> new Extra initialized from a mapping object's
          (key, value) pairs.
        - Extra(iterable) -> new Extra initialized as if via:

          .. code-block:: python

            d = {}
            for k, v in iterable:
                d[k] = v
            Extra(d)

        - Extra(**kwargs) -> new Extra initialized with the name=value
          pairs in the keyword argument list. For example: Extra(one=1, two=2).

    """

    def __init__(self, data):
        self._data = dict(data)

    def __dir__(self):
        return self._data.keys()

    def __eq__(self, obj):
        """x.__eq__(y) <==> x==y."""
        if not isinstance(obj, Extra):
            return False
        if sorted(self._data.keys()) != sorted(obj._data.keys()):
            return False
        for k, v in self._data.items():
            ov = obj._data[k]
            if not isinstance(ov, type(v)):
                return False
            eq = iter_equal if isinstance(v, np.ndarray) else operator.eq
            if not eq(v, ov):
                return False
        return True

    def __ne__(self, obj):
        """x.__ne__(y) <==> x!=y."""
        return not self == obj

    def __getitem__(self, k):
        """x.__getitem__(y) <==> x[y]."""
        return self._data[k]

    def __iter__(self):
        """x.__iter__() <==> iter(x)."""
        return iter(self._data)

    def __len__(self):
        """x.__len__() <==> len(x)."""
        return len(self._data)

    def __getattr__(self, k):
        """x.__getattr__('name') <==> x.name."""
        try:
            return self._data[k]
        except KeyError:
            msg = "'Extra' object has no attribute '{}'".format(k)
            raise AttributeError(msg)

    def __str__(self):
        """x.__str__() <==> repr(x)."""
        return self.to_str()

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        return str(self)

    def to_str(self):
        """Return an string implementation of the extra data."""
        return "Extra({})".format(", ".join(self._data))


# =============================================================================
# DECISION
# =============================================================================

class Decision(object):
    """Represent a result of a Decision Maker method.

    Params
    ------

    decision_maker : decision maker instance.
        The object that make this decision.
    data : skcriteria.Data instance.
        All the information of the alternatives and criteria.
        used by the decision maker to make this decision.
    kernel_ : ndarray.
        indexes of non superated alternatives.
    rank_ : iterable
        rank scores of all the alternatives of the data.
    e_: dict
        extra information of the decision maker.

    """
    def __init__(self, decision_maker, data, kernel_, rank_, e_):
        self._decision_maker = decision_maker
        self._data = data
        self._kernel = kernel_
        self._rank = rank_
        self._e = Extra(e_)

    def _iter_rows(self):
        title = []
        if self._rank is not None:
            title.append("Rank")
        if self._kernel is not None:
            title.append("Kernel")
        for idx, row in enumerate(self._data._iter_rows()):
            if idx == 0:
                extra = title
            else:
                aidx = idx - 1
                extra = []
                if self._rank is not None:
                    extra.append(self._rank[aidx])
                if self._kernel is not None:
                    extra.append("  @" if idx - 1 in self._kernel else "")
            yield row + extra

    def __eq__(self, obj):
        """x.__eq__(y) <==> x == y."""
        return (
            isinstance(obj, Decision) and
            self._decision_maker == obj._decision_maker and
            self._data == obj._data and
            iter_equal(self._kernel, obj._kernel) and
            iter_equal(self._rank, obj._rank) and
            self._e == obj._e)

    def __ne__(self, obj):
        """x.__ne__(y) <==> x != y."""
        return not self == obj

    def __str__(self):
        """x.__str__() <==> str(x)"""
        return "{} - Solution:\n{}".format(
            repr(self._decision_maker)[1: -1], self.to_str())

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        return str(self)

    def _repr_html_(self):
        """Return a html representation for a particular decision."""
        uid = "dec-" + str(uuid.uuid1())
        table = self.to_str(tablefmt="html")
        dm = repr(self._decision_maker)[1: -1]
        return "<div id='{}'><p><b>{} - Solution:</b></p>{}</div>".format(
            uid, dm, table)

    def to_str(self, **params):
        """Return an string implementation of the decision."""
        params.update({
            k: v for k, v in TABULATE_PARAMS.items() if k not in params})
        rows = self._iter_rows()
        return tabulate(rows, **params)

    def as_dict(self):
        """Return an dict implementation of the obect."""
        data = {
            "data": self._data.as_dict(),
            "kernel_": self._kernel,
            "rank_": self._rank, "e_": self._e}
        dm = self._decision_maker.as_dict()
        data.update({"decision_maker": dm})
        return data

    @property
    def decision_maker(self):
        """Decision maker instance tat create this decision."""
        return self._decision_maker

    @property
    def data(self):
        """Data used to create this decision."""
        return self._data

    @property
    def mtx(self):
        """Decision matrix used to create this decision.

        Notes
        -----

        This is a shorcut for ``self.data.mxt``.

        """
        return self._data.mtx

    @property
    def criteria(self):
        """Criteria vector used to create this decision.

        Notes
        -----

        This is a shorcut for ``self.data.criteria``.

        """
        return self._data.criteria

    @property
    def weights(self):
        """Weights used to create this decision.

        Notes
        -----

        This is a shorcut for ``self.data.weights``.

        """
        return self._data.weights

    @property
    def kernel_(self):
        """Kernel of the non-superated alternatives or ``None`` if
        the decision is not :math:`beta` solution.

        """
        return self._kernel

    @property
    def rank_(self):
        """Rank of alternatives or ``None`` if
        the decision is type :math:`beta` solution.

        """
        return self._rank

    @property
    def e_(self):
        """Extra information given by the decision maker"""
        return self._e

    @property
    def best_alternative_(self):
        """Best alternative or ``None`` if
        the decision is type :math:`beta` solution.

        """
        if self._rank is not None:
            return np.argmin(self._rank)

    @property
    def alpha_solution_(self):
        """If is True the decision has a full ranking of alternatives."""
        return self._rank is not None

    @property
    def beta_solution_(self):
        """If is True the decision determines a group of non dominated
        alternatives.

        """
        return self._kernel is not None

    @property
    def gamma_solution_(self):
        """If is True the decision determines the best alternative."""
        return self._rank is not None


# =============================================================================
# DECISION MAKER
# =============================================================================

class DecisionMaker(BaseSolver):

    def make_result(self, data, kernel, rank, extra):
        """Create a new :py:class:`skcriteria.madm.Decision`

        This function receives all the raw data of the DecisionMaker
        and create more usefull object.

        Parameters
        ----------

        data : :py:class:`skcriteria.Data`
            A original data provided to the ``solve()`` method.
        kernel : array_like or None.
            An 1d array_like with *n* elements
            where the every element is the index of the the non-superated
            alternatives; or ``None`` if the method not resolve a kernel.
        rank : array_like or None.
            An 1d array_like with the same elements as alternatives in
            data or None. The i-nth element represent the rank of the
            i-nth alternative.
        extra : :py:class:`dict`
            Extra information about the result.

        Returns
        -------

        :py:class:`skcriteria.madm.Decision`
            A convenient instance containing all the parameters.

        """
        decision = Decision(
            decision_maker=self, data=data,
            kernel_=kernel, rank_=rank, e_=extra)
        return decision
