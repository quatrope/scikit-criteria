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
        repeats = self._repeat
        ama = self._allow_missing_alternatives
        lds = self._last_diff_strategy
        seed = self._seed
        return (
            f"<RankReversalTest1 {dm} repeats={repeats}, "
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

        Parameters
        ----------
        dm : ``skcriteria.core.data.DecisionMatrix``
            The decision matrix from which the maximum possible absolute noises
            of each alternative are to be extracted.
        rank : ``skcriteria.madm.Rank``
            Ranking of alternatives.

        Returns
        -------
        Maximum absolute noise: pandas.DataFrame
            Each row contains the maximum possible absolute noise to worsen
            the current alternative (``mutate``) with respect to the next
            (``mute_next``).

        """
        # TODO: room for improvement: pandas to numpy

        # Here we generate a pandas Series of alternatives in order of ranking
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

    def _mutate_dm(self, *, dm, mutate, alternative_max_abs_noise, random):
        """Create a new decision matrix by replacing a suboptimal alternative \
        with a slightly worse one.

        The algorithm operates as follows:

            - A random uniform noise [0, b] is generated, where b is the
              absolute difference in value of the criterion to be modified
              with respect to the immediately worse alternative.
            - Negative sign is assigned to criteria to be maximized.
            - The noise is applied to the alternative to be worsened
              ('mutated').

        This algorithm is designed in such a way that the 'worsened'
        alternative is not worse than the immediately worse one in the original
        ranking

        Parameters
        ----------
        dm : ``skcriteria.core.data.DecisionMatrix``
            The original decision matrix.
        mutate : str
            The alternative to mutate.
        alternative_max_abs_noise: pandas.Series
            The maximum possible noise with which the alternative to mutate can
            be made worse, without being worse than the immediately worse
            alternative.
        random: `numpy.random.default_rng`
            Random number generator.

        Returns
        -------
        mutated_dm: ``skcriteria.DecisionMatrix``
            Decision matrix with the 'mutate' alternative "worsened".
        noise: ``pandas.Series``
            Noise used to worsen the alternative.

        """
        # TODO: room for improvement: pandas to numpy

        # matrix with alternatives
        df = dm.matrix

        noise = 0  # all noises == 0
        while np.all(noise == 0):  # at least we need one noise > 0
            # calculate the noises without sign
            noise = alternative_max_abs_noise.apply(
                lambda b: random.uniform(0, b)
            )

        # negate when the objective is to maximize
        # onwards the noise is no longer absolute
        noise[dm.maxwhere] *= -1

        # apply the noise
        df.loc[mutate] += noise

        # transform the noised matrix into a dm
        mutated_dm = dm.copy(matrix=df.to_numpy(), dtypes=None)

        return mutated_dm, noise

    def _add_mutation_info_to_rank(
        self, *, rank, mutated, noise, iteration, full_alternatives
    ):
        """Adds information on how an alternative was "worsened" in the \
        decision matrix with respect to the original.

        All aggregated information is included within the ``rrt1`` (Rank
        Reversal Test 1) key in the ``extra_`` attribute.

        """
        # extract the original data
        method = f"{rank.method}+RRT1+{mutated}_{iteration}"
        alternatives = rank.alternatives
        values = rank.values
        extra = dict(rank.extra_.items())

        # we check if the decision_maker did not eliminate any alternatives
        alts_diff = arrset.setxor1d(alternatives, full_alternatives)

        # add the missing alternatives withe the worst ranking value
        if len(alts_diff):
            # All missing alternatives have the maximum ranking + 1
            fill_values = np.full_like(alts_diff, rank.rank_.max() + 1)

            # concatenate the missing alternatives and the new rankings
            alternatives = np.concatenate((alternatives, alts_diff))
            values = np.concatenate((values, fill_values))

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
        patched_rank = RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
        return patched_rank

    def _make_mutations(self, *, dm, orank):
        """Generate all experiments data.

        This method yields all the data needed to run the underlying
        decision-maker with all possible worst suboptimal alternatives.

        Parameters
        ----------
        dm : ``skcriteria.core.data.DecisionMatrix``
            The decision matrix to mutate in every experiment.
        orank : ``skcriteria.madm.Rank``
            The original ranking without mutations.

        Yields
        ------
        iteration: int
            Each suboptimal alternative is worsened `repeat` times. This value
            keeps a count of which of those times the ``mutated`` alternative
            is worsened.
        mutate: str
            The name of the worsened alternative.
        mutated_dm: ``skcriteria.core.data.DecisionMatrix``
            The decision matrix with the suboptimal alternative already
            worsen.
        noise: ``pandas.Series``
            Noise used to worsen the 'mutate' alternative.

        """
        # check the maximum absolute difference between any alternative and
        # the next one in the ranking to establish a worse-limit
        maximum_abs_noises = self._maximum_abs_noises(dm=dm, rank=orank)

        # we repeat the experiment _repeats time
        for iteration in range(self._repeat):
            # we iterate over every sub optimal alternative
            for (mutate, _), alt_max_anoise in maximum_abs_noises.iterrows():
                # create the new matrix with a worse alternative than mutate
                mutated_dm, noise = self._mutate_dm(
                    dm=dm,
                    mutate=mutate,
                    alternative_max_abs_noise=alt_max_anoise,
                    random=self._random,
                )
                yield iteration, mutate, mutated_dm, noise

    def evaluate(self, dm):
        # FIRST THE DATA THAT WILL BE USED IN ALL THE ITERATIONS ==============

        # we need a first reference ranking
        rank = self._dmaker.evaluate(dm)

        # all alternatives to be used to check consistency
        full_alternatives = dm.alternatives

        # Here we create a containers for the rank comparator starting with
        # the original rank
        names, results = ["Original"], [rank]

        # START EXPERIMENTS ===================================================

        for it, mutated, mdm, noise in self._make_mutations(dm=dm, orank=rank):
            # calculate the new rank
            mrank = self._dmaker.evaluate(mdm)

            # add info about the mutation to rhe rank
            patched_mutated_rank = self._add_mutation_info_to_rank(
                rank=mrank,
                mutated=mutated,
                noise=noise,
                iteration=it,
                full_alternatives=full_alternatives,
            )

            # check for missing alternatives
            miss_alts = set(patched_mutated_rank.e_.rrt1.missing_alternatives)
            if self._allow_missing_alternatives is False and miss_alts:
                raise ValueError(
                    f"Missing alternative/s {miss_alts!r} in mutation "
                    f"{mutated!r} of iteration {it}"
                )

            # store the information
            names.append(f"M.{mutated}")
            results.append(patched_mutated_rank)

        # manually creates a new RankComparator
        named_ranks = unique_names(names=names, elements=results)
        return RanksComparator(named_ranks)
