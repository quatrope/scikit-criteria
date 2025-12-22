#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for RankTransitivityChecker with Van Herden dataset."""

import skcriteria as skc
from skcriteria.agg.topsis import TOPSIS
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.preprocessing.scalers import VectorScaler
from skcriteria.ranksrev.rank_transitivity_check import RankTransitivityChecker

# Load the Van Herden dataset (van2021evaluation)
dm = skc.datasets.load_van2021evaluation(windows_size=7)

print("Dataset loaded:")
print(dm)
print()

# Create a normalization pipeline with TOPSIS
topsis_pipe = mkpipe(
    InvertMinimize(),
    VectorScaler(target="matrix"),
    TOPSIS(),
)

print("Pipeline created:")
print(topsis_pipe)
print()

# Create RankTransitivityChecker
trans_checker = RankTransitivityChecker(
    topsis_pipe,
    cycle_removal_strategy="random",
    max_ranks=50,
)

print("RankTransitivityChecker created:")
print(trans_checker)
print()

# Evaluate using the transitivity checker
result = trans_checker.evaluate(dm)

print("Evaluation result:")
print(result)
print()
print("Ranks:")
print(result.rank_)
