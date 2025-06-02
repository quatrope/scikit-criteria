#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Tools for evaluating the stability of MCDA method's best alternative.

According to this criterion, dividing the alternatives by pair should
keep transitivity when grouped again.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

import joblib

import numpy as np

import pandas as pd

from ..agg import RankResult
from ..cmp import RanksComparator
from ..core import SKCMethodABC
from ..utils import Bunch, unique_names


# =============================================================================
# CLASS
# =============================================================================


class TransitivityChecker(SKCMethodABC):
    r""""""

    def __init__(self, dmaker, *, parallel_backend=None, random_state=None):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        # PARALLEL BACKEND
        self._parallel_backend = parallel_backend

        # RANDOM
        self._random_state = np.random.default_rng(random_state)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        name = self.get_method_name()
        dm = repr(self.dmaker)
        repeats = self.repeat
        ama = self._allow_missing_alternatives
        lds = self.last_diff_strategy
        return (
            f"<{name} {dm} repeats={repeats}, "
            f"allow_missing_alternatives={ama} last_diff_strategy={lds!r}>"
        )

    # PROPERTIES ==============================================================

    @property
    def dmaker(self):
        """The MCDA method, or pipeline to evaluate."""
        return self._dmaker

    @property
    def parallel_backend(self):
        return self._parallel_backend

    @property
    def random_state(self):
        """Controls the random state to generate variations in the \
        sub-optimal alternatives."""
        return self._random_state

    # LOGIC ===================================================================

    def _evaluate_pairwise_submatrix(
        decision_matrix, alternative_pair, pipeline
    ):
        """
        Apply the MCDM pipeline to a sub-problem of two alternatives
        """
        sub_dm = decision_matrix.loc[alternative_pair]
        return pipeline.evaluate(sub_dm)

    def evaluate(self, dm):
        """Executes a the transitivity test.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix to be evaluated.

        Returns
        -------
        RanksComparator
            An object containing multiple rankings of the alternatives, with
            information on any changes made to the original decision matrix in
            the `extra_` attribute. Specifically, the `extra_` attribute
            contains a an object in the key `rrt1` that provides
            information on any changes made to the original decision matrix,
            including the the noise applied to worsen any sub-optimal
            alternative.

        """
        # FIRST THE DATA THAT WILL BE USED IN ALL THE ITERATIONS ==============

        # the test configuration
        dmaker = self.dmaker
        parallel_backend = self._parallel_backend
        random = self.random_state

        # all alternatives to be used to check consistency
        full_alternatives = dm.alternatives

        # Pipeline to apply to all pairwise sub-problems
        ws_pipe = mkpipe(
            InvertMinimize(),
            FilterNonDominated(),
            SumScaler(target="weights"),
            VectorScaler(target="matrix"),
            WeightedSumModel(),
        )

        # Load Van 2021 Evaluation Dataset of cryptocurrencies
        dm = skc.datasets.load_van2021evaluation(windows_size=7)

        # Get original ranking
        original_rank = ws_pipe.evaluate(dm)

        # Generate all pairwise combinations of alternatives
        # For n alternatives, creates C(n,2) = n*(n-1)/2 unique sub-problems
        pairwise_combinations = map(list, it.combinations(dm.alternatives, 2))

        # Parallel processing of all pairwise sub-matrices
        # Each resulting sub-matrix has 2 alternatives × k original criteria
        with joblib.Parallel(prefer="processes") as P:
            delayed_evaluation = joblib.delayed(_evaluate_pairwise_submatrix)
            results = P(
                delayed_evaluation(dm, pair, ws_pipe)
                for pair in pairwise_combinations
            )
