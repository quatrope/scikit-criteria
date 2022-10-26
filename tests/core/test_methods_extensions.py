#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.methods

"""


# =============================================================================
# IMPORTS
# =============================================================================

import pytest

from skcriteria import madm
from skcriteria.cmp import RanksComparator
from skcriteria.core import methods
from skcriteria.pipeline import SKCPipeline
from skcriteria.preprocessing import SKCMatrixAndWeightTransformerABC
from skcriteria.preprocessing.filters import (
    Filter,
    SKCArithmeticFilterABC,
    SKCSetFilterABC,
)


# =============================================================================
# TEST UTILITES
# =============================================================================


def _get_subclasses(cls):

    is_abstract = vars(cls).get("_skcriteria_abstract_class", False)

    if not is_abstract and cls.copy == methods.SKCMethodABC.copy:
        yield cls

    for subc in cls.__subclasses__():
        for subsub in _get_subclasses(subc):
            yield subsub


# =============================================================================
# TEST COPY
# =============================================================================


@pytest.mark.run(order=-1)
def test_SLCMethodABC_concrete_subclass_copy():

    # CLASSES FOR THE FAKE PIPELINE
    class _FakeTrans:
        def transform(self):
            pass

        def __eq__(self, o):
            return isinstance(o, _FakeTrans)

    class _FakeDM:
        def evaluate(self):
            pass

        def __eq__(self, o):
            return isinstance(o, _FakeDM)

    steps = [
        ("trans", _FakeTrans()),
        ("dm", _FakeDM()),
    ]

    # ranks for fake RanksComparator
    ranks = [
        ("r0", madm.RankResult("r0", ["a1"], [1], {})),
        ("r1", madm.RankResult("r1", ["a1"], [1], {})),
    ]

    # Some methods need extra parameters.
    extra_parameters_by_type = {
        SKCMatrixAndWeightTransformerABC: {"target": "both"},
        SKCPipeline: {"steps": steps},
        RanksComparator: {"ranks": ranks},
        Filter: {"criteria_filters": {"foo": lambda e: e}},
        SKCArithmeticFilterABC: {"criteria_filters": {"foo": 1}},
        SKCSetFilterABC: {"criteria_filters": {"foo": [1]}},
    }

    for scls in _get_subclasses(methods.SKCMethodABC):

        kwargs = {}
        for cls, extra_params in extra_parameters_by_type.items():
            if issubclass(scls, cls):
                kwargs.update(extra_params)

        original = scls(**kwargs)
        copy = original.copy()

        assert (
            original.get_parameters() == copy.get_parameters()
        ), f"'{scls.__qualname__}' instance not correctly copied."
