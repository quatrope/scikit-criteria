#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Rank reversal test 1"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import numpy.lib.arraysetops as arrset
import pandas as pd

from ...core import SKCMethodABC
from ...madm import RankResult
from ...utils import Bunch, unique_names
from .. import RanksComparator

# =============================================================================
# CLASS
# =============================================================================

_LAST_DIFF_STRATEGIES = {
    "median": np.median,
    "mean": np.mean,
    "average": np.mean,
}


class RankReversalTest1:
    def __init__(
        self,
        dmaker,
        *,
        repeat=1,
        allow_missing_alternatives=False,
        last_diff_strategy="median",
        seed=None,
    ):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        # REPEAT AND ALLOW MISSING ALTERNATIVES
        self._repeat = int(repeat)
        self._allow_missing_alternatives = bool(allow_missing_alternatives)

        # LAST DIFF
        self._last_diff_strategy = (
            _LAST_DIFF_STRATEGIES.get(last_diff_strategy)
            if isinstance(last_diff_strategy, str)
            else last_diff_strategy
        )
        if not callable(self._last_diff_strategy):
            diff_alternatives = ", ".join(_LAST_DIFF_STRATEGIES)
            msg = (
                "'last_diff_strategy' must be a "
                f"'{diff_alternatives}' or a callable"
            )
            raise TypeError(msg)

        # RANDOM
        self._seed = seed
        self._random = np.random.default_rng(seed)

    def __repr__(self):
        dm = repr(self._dmaker)
        if len(dm) > 50:
            dm = dm[:50] + "..." + dm[-1]

        repeats = self._repeat
        ama = self._allow_missing_alternatives
        lds = self._last_diff_strategy
        seed = self._seed
        return (
            f"<RankReversalTest1 {dm} repeats={repeats},  "
            f"allow_missing_alternatives={ama} last_diff_strategy={lds!r} "
            f"seed={seed!r}>"
        )

    def _maximum_abs_noises(self, *, dm, rank):
        """Calculate the absolute difference between the alternatives in order.

        This difference is used as a maximum possible noise to worsen an
        alternative.

        The last alternative in the ranking has no next alternative to compare,
        so the value calculated by ``last_diff_strategy`` is applied as the
        abs difference (the default is the ``np.nanmedian``).

        """
        # TODO: room for improvement by replace all pandas operation
        # into numpy operations

        # here we create a pd.Series of alternatives in rank order
        alts = rank.to_series().sort_values().index.to_numpy()

        # We only need all but the best one.
        not_best = alts[1:]

        # we only need the rows without the best alternative
        not_best_df = dm.matrix.loc[not_best]

        # we need to create the index
        not_best_and_next = dict(zip(not_best, np.roll(not_best, -1)))
        not_best_and_next[not_best[-1]] = np.nan
        not_best_and_next = list(not_best_and_next.items())
        index = pd.MultiIndex.from_tuples(
            not_best_and_next, names=["mutate", "mutate_next"]
        )

        # cleanup
        del alts, not_best, not_best_and_next

        # we create the differences
        maximum_abs_noises = not_best_df.diff(-1).abs()
        maximum_abs_noises.set_index(index, inplace=True)

        # we apply the median as last
        maximum_abs_noises.iloc[-1] = maximum_abs_noises.iloc[:-1].apply(
            self._last_diff_strategy
        )

        return maximum_abs_noises

    def _mutate(self, *, dm, mutate, max_abs_noise, random):
        # TODO: room for improvement by replace all pandas operation
        # into numpy operations

        # matrix to prevent to easy mutation
        df = dm.matrix

        noise = 0  # all noises == 0
        while np.all(noise == 0):  # at least we need one noise > 0
            # calculate the noises withoun sign
            noise = max_abs_noise.apply(lambda b: random.uniform(0, b))

        # negate when the objective is to maximize
        # onwards the noise is no loger absolute
        noise[dm.maxwhere] *= -1

        # apply the noise
        df.loc[mutate] += noise

        # transform the noised matrix into a dm
        mutated_dm = dm.copy(matrix=df.to_numpy(), dtypes=None)

        return mutated_dm, noise

    def _add_mutation_info_to_rank(
        self, *, rank, mutated, noise, iteration, full_alternatives
    ):
        # extract the original data
        method = f"{rank.method}+RRT1+{mutated}_{iteration}"
        alternatives = rank.alternatives
        values = rank.values
        extra = dict(rank.extra_.items())

        alts_diff = arrset.setxor1d(alternatives, full_alternatives)

        if len(alts_diff) and self._allow_missing_alternatives is False:
            raise ValueError(
                f"Missing alternative/s {set(alts_diff)!r} in mutation "
                f"{mutated!r} of iteration {iteration}"
            )
        elif len(alts_diff):
            # TODO MEJORAR cuando hay mas de una alternativa faltante
            import ipdb

            ipdb.set_trace()
            fill = np.full_like(alts_diff, len(full_alternatives))
            alternatives = np.concatenate((alternatives, alts_diff))
            values = np.concatenate((values, fill))

        # patch the new data
        extra["rrt1"] = Bunch(
            "rrt1",
            {
                "mutated": mutated,
                "noise": noise.copy(),
                "missing_alternatives": alts_diff,
                "iteration": iteration,
            },
        )

        # return the new rank result
        return RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def evaluate(self, dm):
        # FIRST THE DATA THAT WILL BE USED IN ALL THE ITERATIONS ==============

        # we need a first reference ranking
        rank = self._dmaker.evaluate(dm)

        # check the maximum absolute difference between any alternative and
        # the next one in the ranking
        maximum_abs_noises = self._maximum_abs_noises(dm=dm, rank=rank)

        # all alternatives to be used to check consistency
        full_alternatives = dm.alternatives

        # Here we create a containers for the rank comparator starting with
        # the original rank
        names, results = ["Original"], [rank]

        # START EXPERIMENTS ===================================================

        # we repeat the experiment _repeats time
        for iteration in range(self._repeat):
            # we iterate over every sub optimal alternative
            for (mutate, _), max_abs_noise in maximum_abs_noises.iterrows():
                # create the new matrix with a worse alternative than mutate
                mutated_dm, noise = self._mutate(
                    dm=dm,
                    mutate=mutate,
                    max_abs_noise=max_abs_noise,
                    random=self._random,
                )

                # calculate the new rank
                mutated_rank = self._dmaker.evaluate(mutated_dm)

                # add info about the mutation to rhe rank
                patched_mutated_rank = self._add_mutation_info_to_rank(
                    rank=mutated_rank,
                    mutated=mutate,
                    noise=noise,
                    iteration=iteration,
                    full_alternatives=full_alternatives,
                )

                # store the information
                names.append(f"M.{mutate}")
                results.append(patched_mutated_rank)

        # manually creates a new RankComparator
        named_ranks = unique_names(names=names, elements=results)
        return RanksComparator(named_ranks)
