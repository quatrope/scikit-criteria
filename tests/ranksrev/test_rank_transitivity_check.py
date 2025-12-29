#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""
Tests for the functionalities in the tranistivity_check file
"""


# =============================================================================
# IMPORTS
# =============================================================================

# import networkx as nx

import numpy as np

import pytest

import skcriteria as skc
from skcriteria.agg.simple import WeightedSumModel
from skcriteria.agg import RankResult
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing.filters import FilterGE, FilterNonDominated
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import SumScaler
from skcriteria.ranksrev.rank_transitivity_check import (
    RankTransitivityChecker,
)
from skcriteria.cmp import RanksComparator, mkrank_cmp


# =============================================================================
# PROPERTIES
# =============================================================================


def test_TransitivityChecker():
    pipe = mkpipe(
        FilterGE({"ROE": 2}),  # Almenos rendir 2%,
        FilterNonDominated(),  # chau dominadas!
        InvertMinimize(),  # no más minimización!
        SumScaler(target="weights"),  # normalizamos los pesos
        SumScaler(target="matrix"),  # normalizamos la matriz
        WeightedSumModel(),  # Función de agregación
    )
    dec = RankTransitivityChecker(pipe, allow_missing_alternatives=True)

    dm = skc.datasets.load_simple_stock_selection()

    result = dec.evaluate(dm)

    # expected = RanksComparator(
    expected = RanksComparator(
        [
            (
                "Original",
                RankResult(
                    method="WeightedSumModel",
                    alternatives=["PE", "JN", "AA", "FX", "MM", "GN"],
                    values=np.array([3, 4, 2, 1, 5, 5]),
                    extra={},
                ),
            ),
            (
                "Recomposition.1",
                RankResult(
                    method="WeightedSumModel",
                    alternatives=["PE", "JN", "AA", "FX", "MM", "GN"],
                    values=np.array([3, 4, 2, 1, 5, 5]),
                    extra={},
                ),
            ),
        ], extra={}
    )

    skc.testing.assert_rcmp_equals(expected, result, skip_extra=True)

    # CLAUDe


# def test_TransitivityChecker_bad_pipe():
#     bad_pipe = "Suffering and pain"
#     with pytest.raises(TypeError) as ex:
#         RankTransitivityChecker(bad_pipe)
#         assert "'dmaker' must implement 'evaluate()' method" in str(ex.value)


# def test_TransitivityChecker_dmaker():
#     trans_checker = RankTransitivityChecker(electre2_pipe)
#     assert trans_checker.dmaker == electre2_pipe


# def test_TransitivityChecker_preferred_parallel_backend_none():
#     trans_checker = RankTransitivityChecker(electre2_pipe)
#     assert trans_checker.preferred_parallel_backend is None


# def test_TransitivityChecker_preferred_parallel_backend():
#     trans_checker = RankTransitivityChecker(
#         electre2_pipe, preferred_parallel_backend=electre2_pipe
#     )
#     assert trans_checker.preferred_parallel_backend == electre2_pipe


# def test_TransitivityChecker_allow_missing_alternatives_default():
#     trans_checker = RankTransitivityChecker(topsis_pipe)
#     assert trans_checker.allow_missing_alternatives is False


# def test_TransitivityChecker_allow_missing_alternatives_True():
#     trans_checker = RankTransitivityChecker(
#         topsis_pipe, allow_missing_alternatives=True
#     )
#     assert trans_checker.allow_missing_alternatives is True


# def test_TransitivityChecker_max_ranks_default():
#     trans_checker = RankTransitivityChecker(electre2_pipe)
#     assert trans_checker.max_ranks == 50


# def test_TransitivityChecker_max_ranks_custom():
#     ranks = 42
#     trans_checker = RankTransitivityChecker(electre2_pipe, max_ranks=ranks)
#     assert trans_checker.max_ranks == ranks


# def test_TransitivityChecker_max_ranks_zero():
#     ranks = 0
#     with pytest.raises(ValueError):
#         RankTransitivityChecker(electre2_pipe, max_ranks=ranks)


# def test_TransitivityChecker_n_jobs_default():
#     trans_checker = RankTransitivityChecker(electre2_pipe)
#     assert trans_checker.n_jobs is None


# def test_TransitivityChecker_n_jobs_custom():
#     jobs = 42
#     trans_checker = RankTransitivityChecker(electre2_pipe, n_jobs=jobs)
#     assert trans_checker.n_jobs == jobs


# def test_RankTransitivityChecker_parallel_backend_deprecation():
#     with pytest.raises(ValueError):
#         RankTransitivityChecker(
#             electre2_pipe,
#             parallel_backend="foo",
#             preferred_parallel_backend="bar",
#         )

#     with pytest.warns(SKCriteriaDeprecationWarning):
#         checker = RankTransitivityChecker(
#             electre2_pipe, parallel_backend="foo"
#         )
#         assert checker.parallel_backend == "foo"


# # =============================================================================
# # TEST MISSING ALTERNATIVES
# # =============================================================================


# def test_TransitivityCheck_missing_alternative_forbidden():
#     dm = skc.datasets.load_simple_stock_selection()
#     trans_check = RankTransitivityChecker(
#         topsis_pipe, allow_missing_alternatives=False
#     )
#     with pytest.raises(ValueError):
#         trans_check.evaluate(dm=dm)


# def test_TransitivityCheck_missing_alternative():
#     dm = skc.datasets.load_simple_stock_selection()
#     trans_check = RankTransitivityChecker(
#         topsis_pipe, allow_missing_alternatives=True
#     )
#     result = trans_check.evaluate(dm=dm)

#     _, rank = result.ranks[1]

#     np.testing.assert_array_equal(
#         rank.e_.transitivity_check.missing_alternatives, ["FX", "MM"]
#     )

#     assert rank.to_series()["FX"] == 5
#     assert rank.to_series()["MM"] == 5
#     assert rank.has_ties_


# # ============================================================================
# # TEST CRITERIA
# # =============================================================================


# def test_TransitivityCheck_test_criterion_2_pass():
#     dm = skc.datasets.load_van2021evaluation(windows_size=7)
#     trans_check = RankTransitivityChecker(topsis_pipe)
#     rank_comparator = trans_check.evaluate(dm=dm)
#     orank = topsis_pipe.evaluate(dm)
#     test_criterion_2 = trans_check._test_criterion_2(dm, orank, None, None)[0]
#     assert rank_comparator._extra.transitivity_break_rate == 0
#     assert rank_comparator._extra.test_criterion_2
#     assert test_criterion_2


# def test_TransitivityCheck_test_criterion_2_fail():
#     dm = skc.datasets.load_van2021evaluation(windows_size=7)
#     trans_check = RankTransitivityChecker(topsis_pipe_moora)
#     rank_comparator = trans_check.evaluate(dm=dm)
#     orank = topsis_pipe.evaluate(dm)
#     test_criterion_2 = trans_check._test_criterion_2(dm, orank, None, None)[0]
#     assert rank_comparator._extra.transitivity_break_rate > 0
#     assert not rank_comparator._extra.test_criterion_2
#     assert not test_criterion_2


# def test_TransitivityCheck_test_criterion_3_pass():
#     dm = skc.datasets.load_van2021evaluation(windows_size=7)
#     trans_check = RankTransitivityChecker(topsis_pipe)
#     rank_comparator = trans_check.evaluate(dm=dm)
#     assert rank_comparator._extra.test_criterion_3


# def test_TransitivityCheck_test_criterion_3_fail():
#     dm = skc.datasets.load_van2021evaluation(windows_size=7)
#     trans_check = RankTransitivityChecker(topsis_pipe_matrix_scaler)
#     rank_comparator = trans_check.evaluate(dm=dm)
#     assert not rank_comparator._extra.test_criterion_3
