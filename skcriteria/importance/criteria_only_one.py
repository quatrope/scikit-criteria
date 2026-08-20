#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Only-one criteria importance checker.

This module measures the importance of each criterion of a decision
problem by keeping *only* that criterion (dropping every other one), with
its weight set to 1, evaluating ``dmaker`` on that single-criterion
sub-problem, and comparing the resulting ranking against a reference
ranking computed with all the criteria. It is the complement of
:class:`~skcriteria.importance.criteria_leave_one_out.\
CriteriaLeaveOneOutChecker` at the opposite end of the coalition chain.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import joblib

import numpy as np

from ..agg import RankResult
from ..cmp import RanksComparator
from ..core import SKCMethodABC
from ..preprocessing.scalers import SumScaler
from ..utils import unique_names

# =============================================================================
# CONSTANTS
# =============================================================================

_VALID_METRICS = frozenset(["footrule", "kendall"])

#: Renormalizes weights to sum 1, used for the reference problem. Each
#: single-criterion sub-problem instead sets its only weight to 1
#: directly (see `_evaluate_only_one_criterion`), which is equivalent
#: but avoids a 0/0 if the original weight happened to be 0.
_WEIGHT_SCALER = SumScaler(target="weights")


# =============================================================================
# FUNCTIONS
# =============================================================================


def _evaluate_only_one_criterion(dmaker, dm, criterion):
    """Evaluate ``dmaker`` on ``dm`` restricted to just ``criterion``.

    Defined at module level (instead of as a method) so it can be safely
    parallelized with ``joblib``.

    Parameters
    ----------
    dmaker : object
        Decision maker instance implementing ``evaluate(dm)``.
    dm : DecisionMatrix
        The (already weight-normalized) full decision matrix.
    criterion : str
        Name of the only criterion to keep.

    Returns
    -------
    criterion : str
        The same ``criterion`` received, to identify the result once
        collected back.
    RankResult
        Ranking obtained by evaluating ``dmaker`` on the sub-problem.

    """
    dm_sub = dm[criterion].replace(weights=[1])
    rank_sub = dmaker.evaluate(dm_sub)
    return criterion, rank_sub


# =============================================================================
# CLASS
# =============================================================================


class CriteriaOnlyOneChecker(SKCMethodABC):
    r"""Only-one importance of each decision-matrix criterion.

    For every criterion :math:`i` in the decision matrix, this checker
    builds a sub-problem :math:`\{i\}` keeping *only* that criterion
    (weight set to 1), evaluates ``dmaker`` on that sub-problem, and
    compares the resulting ranking against the reference ranking
    (evaluated with all the criteria, :math:`N`) using a pairwise
    ranking-similarity metric.

    Parameters
    ----------
    dmaker : object
        Decision maker instance that must implement an ``evaluate(dm)``
        method returning a ``skcriteria.agg.RankResult``. This represents
        the MCDA method or pipeline whose sensitivity to each criterion is
        being measured.

    metric : str, default ``"footrule"``
        Ranking-similarity metric used to compare a single-criterion
        ranking against the reference ranking. One of:

        - ``"footrule"``: normalized Spearman footrule similarity, computed
          with :meth:`skcriteria.cmp.RanksComparator.footrule_similarity`.
        - ``"kendall"``: Kendall tau rank correlation, computed with
          :meth:`skcriteria.cmp.RanksComparator.corr` using
          ``method="kendall"``.

    untied : bool, default ``False``
        Forwarded as-is to ``RanksComparator.footrule_similarity``/
        ``RanksComparator.corr``. If ``True`` and any ranking (reference or
        single-criterion) has ties, ``RankResult.untied_rank_`` is used to
        assign each alternative a single ranked order before computing the
        pairwise metric. If ``False``, the rankings are used as they are.

    allow_missing_alternatives : bool, default ``True``
        ``dmaker`` can somehow return rankings with fewer alternatives than
        the original decision matrix (using a pipeline that implements a
        filter, for example). If ``True``, any alternative missing from a
        ranking (the reference or any single-criterion ranking) is added
        back with a value equal to the worst rank obtained in that ranking
        plus 1. If ``False``, a ranking missing an alternative raises a
        ``ValueError``.

    preferred_parallel_backend : str or None, default ``None``
        Soft hint passed as ``joblib.Parallel(prefer=...)`` to parallelize
        the evaluation of the single-criterion sub-problems (one per
        criterion). One of ``"threads"``, ``"processes"``, or ``None`` for
        joblib's default behavior.

    n_jobs : int or None, default ``None``
        Number of parallel jobs used to evaluate the single-criterion
        sub-problems. ``None`` means sequential execution (joblib's
        default of a single job); ``-1`` uses all available processors.

    Raises
    ------
    TypeError
        If ``dmaker`` doesn't implement the required ``evaluate()`` method.
    ValueError
        If ``metric`` is not ``"footrule"`` or ``"kendall"``.

    Notes
    -----
    This checker uses the same pairwise-comparison machinery as
    :class:`~skcriteria.importance.criteria_leave_one_out.\
CriteriaLeaveOneOutChecker`,
    reusing :meth:`skcriteria.cmp.RanksComparator.footrule_similarity` or
    :meth:`skcriteria.cmp.RanksComparator.corr` instead of an ad hoc
    distance function, but on the opposite sub-problem: instead of
    dropping one criterion from the full set :math:`N`, it keeps *only*
    one criterion from the empty set :math:`\emptyset`. The reported score
    is ``1 - similarity`` (or its Kendall-tau rescaling) between the
    single-criterion ranking and the reference ranking, exactly as in
    ``CriteriaLeaveOneOutChecker``; it should be read as *how different*
    the ranking is when using only that criterion, not as a measure of how
    well that criterion alone reproduces the reference ranking.

    """

    _skcriteria_dm_type = "sensitivity_importance"
    _skcriteria_parameters = [
        "dmaker",
        "metric",
        "untied",
        "allow_missing_alternatives",
        "preferred_parallel_backend",
        "n_jobs",
    ]

    def __init__(
        self,
        dmaker,
        *,
        metric="footrule",
        untied=False,
        allow_missing_alternatives=True,
        preferred_parallel_backend=None,
        n_jobs=None,
    ):
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError("'dmaker' must implement 'evaluate()' method")
        self._dmaker = dmaker

        if metric not in _VALID_METRICS:
            valid = ", ".join(sorted(_VALID_METRICS))
            raise ValueError(
                f"'metric' must be one of ({valid}). Found {metric!r}"
            )
        self._metric = metric

        self._untied = bool(untied)

        self._allow_missing_alternatives = bool(allow_missing_alternatives)

        self._preferred_parallel_backend = preferred_parallel_backend
        self._n_jobs = None if n_jobs is None else int(n_jobs)

    # PROPERTIES ==============================================================

    @property
    def dmaker(self):
        """The MCDA method, or pipeline, whose sensitivity is measured."""
        return self._dmaker

    @property
    def metric(self):
        """Ranking-similarity metric used to score each criterion's \
        importance."""
        return self._metric

    @property
    def untied(self):
        """If ``True``, ties are broken with \
        ``RankResult.untied_rank_`` before computing the pairwise metric \
        (same convention as ``RanksComparator``)."""
        return self._untied

    @property
    def allow_missing_alternatives(self):
        """True if single-criterion rankings are allowed to be missing \
        alternatives with respect to the original decision matrix."""
        return self._allow_missing_alternatives

    @property
    def preferred_parallel_backend(self):
        """Backend used to parallelize the evaluation of the \
        single-criterion sub-problems."""
        return self._preferred_parallel_backend

    @property
    def n_jobs(self):
        """Number of parallel jobs used to evaluate the single-criterion \
        sub-problems."""
        return self._n_jobs

    # INTERNALS ===============================================================

    def _patch_missing_alternatives(self, *, rank, full_alternatives, where):
        """Fill any alternative missing from ``rank`` (with respect to \
        ``full_alternatives``) with the worst rank + 1.

        Same convention used by
        ``skcriteria.ranksrev.rank_invariant_check.RankInvariantChecker``.

        """
        method = str(rank.method)
        alternatives = rank.alternatives.copy()
        values = rank.values.copy()
        extra = dict(rank.extra_.items())

        alts_diff = np.setxor1d(alternatives, full_alternatives)
        missing_alternatives = np.array([], dtype=full_alternatives.dtype)

        if len(alts_diff):
            if not self._allow_missing_alternatives:
                missing_alts = set(alts_diff)
                raise ValueError(
                    f"Missing alternative/s {missing_alts!r} in {where}"
                )

            missing_alternatives = alts_diff

            # add missing alternatives with the worst ranking + 1; the fill
            # array must match `values`' dtype (not `alts_diff`'s), or the
            # concatenation below silently upcasts the ranks to `object`,
            # which later breaks scipy.spatial.distance.pdist downstream
            fill_values = np.full(
                len(alts_diff), rank.rank_.max() + 1, dtype=values.dtype
            )
            alternatives = np.concatenate((alternatives, alts_diff))
            values = np.concatenate((values, fill_values))

            # restore the original order of alternatives
            order = {alt: i for i, alt in enumerate(full_alternatives)}
            indices = np.argsort([order[alt] for alt in alternatives])
            alternatives = alternatives[indices]
            values = values[indices]

        patched_rank = RankResult(
            method=method,
            alternatives=alternatives,
            values=values,
            extra=extra,
        )
        return patched_rank, missing_alternatives

    def _importance_score(self, named_ranks):
        """Importance of every ranking in ``named_ranks`` vs ``"reference"``.

        Returned as a ``pandas.Series`` indexed by ranking name, always
        bounded in ``[0, 1]``. Builds the temporary ``RanksComparator``
        needed to compute the pairwise metric and reuses its own
        vectorized methods (computing the full matrix once), indexes the
        ``"reference"`` row out of it, and returns how much the ranking
        changed, not how similar it stayed. ``"reference"`` itself gets
        an importance of 0.

        ``footrule_similarity()`` is already bounded in ``[0, 1]``, so its
        complement (``1 - similarity``) is used directly. Kendall's tau is
        a correlation bounded in ``[-1, 1]`` instead, so its complement is
        rescaled by half (``(1 - correlation) / 2``) to keep importance in
        the same ``[0, 1]`` range regardless of ``metric``.
        """
        rcmp = RanksComparator(named_ranks, extra={})
        if self._metric == "footrule":
            similarity_matrix = rcmp.footrule_similarity(untied=self._untied)
            similarity = similarity_matrix.loc["reference"]
            importance = 1.0 - similarity
        else:
            correlation_matrix = rcmp.corr(
                method="kendall", untied=self._untied
            )
            correlation = correlation_matrix.loc["reference"]
            importance = (1.0 - correlation) / 2.0

        importance.name = "Importance"

        return importance

    # LOGIC ===================================================================

    def evaluate(self, dm):
        r"""Execute the only-one importance test.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix to be evaluated. Must have at least 2
            criteria.

        Returns
        -------
        RanksComparator
            An object containing the reference ranking (named
            ``"reference"``) plus one ranking per criterion (named
            ``"OO-{criterion}"``), obtained by evaluating ``dmaker``
            using only that criterion, with its weight set to 1. The
            ``extra_`` attribute contains:

            - ``metric``: the metric used (``"footrule"`` or ``"kendall"``).
            - ``importance``: a ``pandas.Series``, indexed by ranking
              name (``"reference"`` and every ``"OO-{criterion}"``),
              always bounded in ``[0, 1]`` (0 means the single-criterion
              ranking is identical to the reference; 1 means the maximum
              possible difference for ``metric``; ``"reference"`` itself
              is always 0). As explained in the class Notes, this reuses
              the same ``1 - similarity`` formula as
              ``CriteriaLeaveOneOutChecker``, so it measures *how
              different* the ranking is, not how well the criterion alone
              reproduces the reference.

        Raises
        ------
        ValueError
            If ``dm`` has fewer than 2 criteria, or if
            ``allow_missing_alternatives`` is ``False`` and some ranking is
            missing an alternative.

        """
        criteria = list(dm.criteria)
        if len(criteria) < 2:
            raise ValueError(
                "CriteriaOnlyOneChecker requires at least 2 criteria"
            )

        # normalize the weights of the full problem first, just in case,
        # so the reference ranking follows the same "weights sum to 1"
        # convention applied to every single-criterion sub-problem
        dm = _WEIGHT_SCALER.transform(dm)

        full_alternatives = np.array(dm.alternatives)

        # reference ranking, using all the criteria
        rank_full = self._dmaker.evaluate(dm)
        patched_full, _ = self._patch_missing_alternatives(
            rank=rank_full,
            full_alternatives=full_alternatives,
            where="reference",
        )

        names = ["reference"]
        results = [patched_full]

        # one single-criterion ranking per criterion, possibly in parallel;
        # the sub-problem is built and evaluated by a module-level function
        # so it can be safely handed off to joblib workers
        dmaker = self._dmaker
        with joblib.Parallel(
            n_jobs=self._n_jobs, prefer=self._preferred_parallel_backend
        ) as parallel:
            delayed_evaluation = joblib.delayed(_evaluate_only_one_criterion)
            only_one_results = parallel(
                delayed_evaluation(dmaker, dm, criterion)
                for criterion in criteria
            )

        # patching for missing alternatives needs 'self', so it is applied
        # sequentially once the (possibly parallel) evaluations come back
        for criterion, rank_sub in only_one_results:
            patched_sub, _ = self._patch_missing_alternatives(
                rank=rank_sub,
                full_alternatives=full_alternatives,
                where=(
                    "the only-one ranking using criterion " f"{criterion!r}"
                ),
            )

            names.append(f"OO-{criterion}")
            results.append(patched_sub)

        # compute the importance-to-reference score once, over a temporary
        # RanksComparator holding every ranking
        named_ranks = unique_names(names=names, elements=results)
        importance_scores = self._importance_score(named_ranks)

        extra = {
            "metric": self._metric,
            "importance": importance_scores,
        }

        return RanksComparator(named_ranks, extra=extra)
