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

According to this criterion, the best alternative identified by the method
should remain unchanged when a non-optimal alternative is replaced by a
worse alternative, provided that the relative importance of each decision
criterion remains the same.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

from ..agg import RankResult
from ..cmp.ranks_cmp import RanksComparator
from ..core import SKCMethodABC
from ..utils import Bunch, unique_names

# =============================================================================
# CONSTANT
# =============================================================================

_LAST_DIFF_STRATEGIES = {
    "median": np.median,
    "mean": np.mean,
}


# =============================================================================
# CLASS
# =============================================================================


class RankInvariantChecker(SKCMethodABC):
    r"""Stability evaluator of MCDA method's best alternative.

    According to this criterion, the best alternative identified by the method
    should remain unchanged when a non-optimal alternative is replaced by a
    worse alternative, provided that the relative importance of each decision
    criterion remains the same.

    To illustrate, suppose that the MCDA method has ranked a set of
    alternatives, and one of the alternatives, :math:`A_j`, is replaced by
    another alternative, :math:`A_j'`, which is less desirable than Ak. The
    MCDA method should still identify the same best alternative when the
    alternatives are re-ranked using the same method. Furthermore, the relative
    rankings of the remaining alternatives that were not changed should also
    remain the same.

    The current implementation worsens each non-optimal alternative ``repeat``
    times, and stores each resulting output in a collection for comparison with
    the reference ranking. In essence, the test is run once for each suboptimal
    alternative.

    This class assumes that there is another suboptimal alternative :math:`A_j`
    that is just the next worst alternative to :math:`A_k`, so that
    :math:`A_k \succ A_j`. Then it generates a mutation :math:`A_k'` such that
    :math:`A_k'` is worse than :math:`A_k` but still better than :math:`A_j`
    (:math:`A_k \succ A_k' \succ A_j`). In the case that the worst alternative
    is reached, its degradation is limited by default  with respect to the
    median of all  limits of the previous alternatives mutations, in order not
    to break he  distribution of each criterion.

    Parameters
    ----------
    dmaker: Decision maker - must implement the ``evaluate()`` method
        The MCDA method, or pipeline to evaluate.

    repeat: int, default = 1
        How many times to mutate each suboptimal alternative.

        The total number of rankings returned by this method is given by the
        number of alternatives in the decision matrix minus one multiplied by
        ``repeat``.

    allow_missing_alternatives: bool, default = False
        ``dmaker`` can somehow return rankings with fewer alternatives than the
        original ones (using a pipeline that implements a filter, for example).
        By setting this parameter to ``True``, the invariance test allows for
        missing alternatives in a ranking to be added with a value of the
        maximum value of the ranking obtained + 1.

        On the other hand, if the value is ``False``, when a ranking is missing
        an alternative, the test will fail with a ``ValueError``.

        If more than one alternative is removed, all of them are added
        with the same value

    last_diff_strategy: str or callable (default: "median").
        True if any mutation is allowed that does not possess all the
        alternatives of the original decision matrix.

    random_state: int, numpy.random.default_rng or None (default: None)
        Controls the random state to generate variations in the sub-optimal
        alternatives.

    """

    _skcriteria_dm_type = "rank_reversal"
    _skcriteria_parameters = [
        "dmaker",
        "repeat",
        "allow_missing_alternatives",
        "last_diff_strategy",
        "random_state",
    ]

    def __init__(
        self,
        dmaker,
        *,
        repeat=1,
        allow_missing_alternatives=False,
        last_diff_strategy="median",
        random_state=None,
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
    def repeat(self):
        """How many times to mutate each suboptimal alternative."""
        return self._repeat

    @property
    def allow_missing_alternatives(self):
        """True if any mutation is allowed that does not possess all the \
        alternatives of the original decision matrix."""
        return self._allow_missing_alternatives

    @property
    def last_diff_strategy(self):
        """Since the least preferred alternative has no lower bound (since \
        there is nothing immediately below it), this function calculates a \
        limit ceiling based on the bounds of all the other suboptimal \
        alternatives."""
        return self._last_diff_strategy

    @property
    def random_state(self):
        """Controls the random state to generate variations in the \
        sub-optimal alternatives."""
        return self._random_state

    # LOGIC ===================================================================

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
        rank : ``skcriteria.agg.Rank``
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
        alts = rank.to_series().sort_values().index.to_numpy(copy=True)

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
            self.last_diff_strategy
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
        alternative is not worse than the immediately worse one in the
        reference ranking.

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
        # we need all the matrix as float for the noise
        df = dm.matrix.astype(float)

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
        mutated_dm = dm.replace(matrix=df.to_numpy(copy=True), dtypes=None)

        return mutated_dm, noise

    def _generate_mutations(self, *, dm, rrank, repeat, random):
        """Generate all experiments data.

        This method yields all the data needed to run the underlying
        decision-maker with all possible worst suboptimal alternatives.

        Parameters
        ----------
        dm : ``skcriteria.core.data.DecisionMatrix``
            The decision matrix to mutate in every experiment.
        rrank : ``skcriteria.agg.Rank``
            The reference ranking without mutations.
        repeat : int
            How many times an suboptimal alternative must be mutated.
        random: `numpy.random.default_rng`
            Random number generator.

        Yields
        ------
        iteration: int
            Each suboptimal alternative is worsened `repeat` times. This value
            keeps a count of which of those times the ``mutated`` alternative
            is worsened.
        mutates: str
            The name of the worsened alternative.
        mutated_dm: ``skcriteria.core.data.DecisionMatrix``
            The decision matrix with the suboptimal alternative already
            worsen.
        noise: ``pandas.Series``
            Noise used to worsen the 'mutated' alternative.

        """
        # check the maximum absolute difference between any alternative and
        # the next one in the ranking to establish a worse-limit
        maximum_abs_noises = self._maximum_abs_noises(dm=dm, rank=rrank)

        # we repeat the experiment _repeats time
        for iteration in range(repeat):
            # we iterate over every sub optimal alternative
            for (mutated, _), alt_max_anoise in maximum_abs_noises.iterrows():
                # create the new matrix with a worse alternative than mutate
                mutated_dm, noise = self._mutate_dm(
                    dm=dm,
                    mutate=mutated,
                    alternative_max_abs_noise=alt_max_anoise,
                    random=random,
                )
                yield iteration, mutated, mutated_dm, noise

    def _add_mutation_info_to_rank(
        self,
        *,
        rank,
        iteration,
        mutated,
        noise,
        full_alternatives,
        allow_missing_alternatives,
    ):
        """Adds information on how an alternative was "worsened" in the \
        decision matrix with respect to the original.

        All aggregated information is included within the ``rank_inv_check``
        key in the ``extra_`` attribute.

        Parameters
        ----------
        rank: ``skcriteria.agg.Rank``
            Ranking to which you want to add information about the executed
            test.
        iteration: int
            Each suboptimal alternative is worsened `repeat` times. This value
            keeps a count of which of those times the ``mutated`` alternative
            is worsened.
        mutated: str
            The name of the worsened alternative.
        noise: ``pandas.Series``
            Noise used to worsen the 'mutated' alternative.
        full_alternatives: array-like
            The full list of alternatives in the original decision matrix.
        allow_missing_alternatives: bool, default ``True``
            If the value is ``False``, when a ranking is missing
            an alternative, the test will fail with a ``ValueError``,
            and if True the missing alternatives are added at the end of the
            ranking all with a value $R_max$ + 1, where $R_max$ is the maximum
            ranking obtained by the alternatives that were not eliminated.

        Returns
        -------
        patched_rank : ``skcriteria.agg.Rank``
            Ranking with all the information about the worsened alternative and
            the rank reversal test added to the `extra_.rank_inv_check`
            attribute.

        """
        # extract the original data
        method = str(rank.method)
        alternatives = rank.alternatives.copy()
        values = rank.values.copy()
        extra = dict(rank.extra_.items())

        # we check if the decision_maker did not eliminate any alternatives
        alts_diff = np.setxor1d(alternatives, full_alternatives)
        has_missing_alternatives = len(alts_diff) > 0

        # check for the missing alternatives
        if has_missing_alternatives:
            # if a missing alternative are not allowed must raise an error
            if not allow_missing_alternatives:
                missing_alts = set(alts_diff)
                where_error = (
                    f"mutation {mutated!r} of iteration {iteration}"
                    if mutated is not None
                    else "Reference run"
                )
                raise ValueError(
                    f"Missing alternative/s {missing_alts!r} in {where_error}"
                )

            # add missing alternatives with the  worst ranking + 1
            fill_values = np.full_like(alts_diff, rank.rank_.max() + 1)

            # concatenate the missing alternatives and the new rankings
            alternatives = np.concatenate((alternatives, alts_diff))
            values = np.concatenate((values, fill_values))

        # change the method name if this is part of a mutation
        if (mutated, iteration) != (None, None):
            method = f"{method}+RInvCheck+{mutated}_{iteration}"
            noise = noise.copy()

        # patch the new data
        extra["rank_inv_check"] = Bunch(
            "rank_inv_check",
            {
                "iteration": iteration,
                "mutated": mutated,
                "noise": noise,
                "missing_alternatives": alts_diff,
            },
        )

        extra["rrt1"] = extra["rank_inv_check"]

        # return the new rank result
        patched_rank = RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
        return patched_rank

    def evaluate(self, dm):
        """Executes a the invariance test.

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
            contains a an object in the key `rank_inv_check` that provides
            information on any changes made to the original decision matrix,
            including the the noise applied to worsen any sub-optimal
            alternative.

        """
        # FIRST THE DATA THAT WILL BE USED IN ALL THE ITERATIONS ==============

        # the test configuration
        dmaker = self.dmaker
        allow_missing_alternatives = self.allow_missing_alternatives
        repeat = self.repeat
        random = self.random_state

        # all alternatives to be used to check consistency
        full_alternatives = dm.alternatives

        # we need a first reference ranking
        rrank = dmaker.evaluate(dm)
        patched_rrank = self._add_mutation_info_to_rank(
            rank=rrank,
            mutated=None,
            noise=None,
            iteration=None,
            full_alternatives=full_alternatives,
            allow_missing_alternatives=allow_missing_alternatives,
        )

        # Here we create a containers for the rank comparator starting with
        # the reference rank
        names, results = ["Reference"], [patched_rrank]

        # START EXPERIMENTS ===================================================
        mutants_generator = self._generate_mutations(
            dm=dm,
            rrank=patched_rrank,
            repeat=repeat,
            random=random,
        )
        for it, mutated, mdm, noise in mutants_generator:
            # calculate the new rank
            mrank = dmaker.evaluate(mdm)

            # add info about the mutation to rhe rank
            patched_mrank = self._add_mutation_info_to_rank(
                rank=mrank,
                mutated=mutated,
                noise=noise,
                iteration=it,
                full_alternatives=full_alternatives,
                allow_missing_alternatives=allow_missing_alternatives,
            )

            # store the information
            names.append(f"M.{mutated}")
            results.append(patched_mrank)

        # manually creates a new RankComparator
        named_ranks = unique_names(names=names, elements=results)
        return RanksComparator(named_ranks, extra={})
