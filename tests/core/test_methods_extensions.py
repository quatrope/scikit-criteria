#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.methods

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skcriteria import agg
from skcriteria.cmp import RankInvariantChecker, RanksComparator
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


def _parameters_diff(original, copy):
    diff = set()
    for k in list(original) + list(copy):
        if k not in original or k not in copy:
            diff.add(k)
        else:
            ov, cv = original[k], copy[k]

            # mayb we need to compare another thing
            if isinstance(ov, np.random.Generator):
                ov = ov.bit_generator.state
                cv = cv.bit_generator.state

            # te comparison!
            if ov != cv:
                diff.add(k)


# =============================================================================
# TEST COPY
# =============================================================================


@pytest.mark.run(order=-1)
@pytest.mark.skip
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
        ("r0", agg.RankResult("r0", ["a1"], [1], {})),
        ("r1", agg.RankResult("r1", ["a1"], [1], {})),
    ]

    # Some methods need extra parameters.
    extra_parameters_by_type = {
        SKCMatrixAndWeightTransformerABC: {"target": "both"},
        SKCPipeline: {"steps": steps},
        RanksComparator: {"ranks": ranks},
        Filter: {"criteria_filters": {"foo": lambda e: e}},
        SKCArithmeticFilterABC: {"criteria_filters": {"foo": 1}},
        SKCSetFilterABC: {"criteria_filters": {"foo": [1]}},
        RankInvariantChecker: {"dmaker": _FakeDM()},
    }

    for scls in _get_subclasses(methods.SKCMethodABC):
        kwargs = {}
        for cls, extra_params in extra_parameters_by_type.items():
            if issubclass(scls, cls):
                kwargs.update(extra_params)

        original = scls(**kwargs)
        copy = original.copy()

        poriginal = original.get_parameters()
        pcopy = copy.get_parameters()

        diff = _parameters_diff(poriginal, pcopy)
        if diff:
            pytest.fail(
                f"'{scls.__qualname__}' instance incorrectly copied. {diff!r}"
            )
