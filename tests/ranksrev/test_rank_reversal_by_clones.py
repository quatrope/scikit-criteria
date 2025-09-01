#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

"""Test for RankClonesChecker."""

import skcriteria as skc
from skcriteria.agg.topsis import TOPSIS
from skcriteria.cmp import RanksComparator
from skcriteria.ranksrev.rank_reversal_by_clones import RankClonesChecker


def test_RankClonesChecker_evaluate():
    """Test the evaluate method of the RankClonesChecker."""    
    # 1. Create a base decision matrix
    dm = skc.datasets.load_simple_stock_selection()

    # 2. Define the decision maker
    dmaker = TOPSIS()

    # 3. Instantiate the checker
    checker = RankClonesChecker(dmaker)

    # 4. Evaluate the decision matrix
    result = checker.evaluate(dm)

    # 5. Assert the result
    assert isinstance(result, RanksComparator)

    named_ranks = dict(result.named_ranks)

    original_rank = named_ranks["Reference"]
    original_best = original_rank.alternatives[0]

    for name, rank in named_ranks.items():
        if name == "Reference":
            assert rank.extra_.rank_clone_check.cloned_alternative is None
            continue

        assert rank.extra_.rank_clone_check.cloned_alternative is not None
        cloned_best = rank.alternatives[0]
        assert original_best == cloned_best
