#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Common machinery for sensitivity-based criteria-importance checkers.

A criteria-importance checker builds one sub-problem per criterion (how
the sub-problem is built is what actually differentiates one checker from
another), evaluates the same ``dmaker`` on it, and compares the resulting
ranking against a reference ranking (evaluated with all the criteria)
using a pairwise ranking-similarity metric reused from
:class:`skcriteria.cmp.RanksComparator`.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import abc

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

#: Renormalizes weights to sum 1, used for the reference problem.
_WEIGHT_SCALER = SumScaler(target="weights")


# =============================================================================
# CLASS
# =============================================================================


class CriteriaImportanceABC(SKCMethodABC):
    """Abstract base for sensitivity-based criteria-importance checkers.

    Subclasses only need to implement :meth:`_evaluate_subproblem`, which
    builds a single criterion's sub-problem, evaluates ``self.dmaker`` on
    it, and returns ``(rank_name, RankResult, extra)``. Everything else --
    parameter validation, the reference ranking, parallelizing the
    per-criterion sub-problems with ``joblib``, patching rankings with
    missing alternatives, and scoring importance against the reference --
    is handled once here.

    Parameters
    ----------
    dmaker : object
        Decision maker instance that must implement an ``evaluate(dm)``
        method returning a ``skcriteria.agg.RankResult``. This represents
        the MCDA method or pipeline whose sensitivity to each criterion is
        being measured.

    metric : str, default ``"footrule"``
        Ranking-similarity metric used to compare a sub-problem ranking
        against the reference ranking. One of:

        - ``"footrule"``: normalized Spearman footrule similarity, computed
          with :meth:`skcriteria.cmp.RanksComparator.footrule_similarity`.
        - ``"kendall"``: Kendall tau rank correlation, computed with
          :meth:`skcriteria.cmp.RanksComparator.corr` using
          ``method="kendall"``.

    untied : bool, default ``False``
        Forwarded as-is to ``RanksComparator.footrule_similarity``/
        ``RanksComparator.corr``. If ``True`` and any ranking (reference or
        sub-problem) has ties, ``RankResult.untied_rank_`` is used to
        assign each alternative a single ranked order before computing the
        pairwise metric. If ``False``, the rankings are used as they are.

    allow_missing_alternatives : bool, default ``True``
        ``dmaker`` can somehow return rankings with fewer alternatives than
        the original decision matrix (using a pipeline that implements a
        filter, for example). If ``True``, any alternative missing from a
        ranking (the reference or any sub-problem ranking) is added back
        with a value equal to the worst rank obtained in that ranking plus
        1. If ``False``, a ranking missing an alternative raises a
        ``ValueError``.

    preferred_parallel_backend : str or None, default ``None``
        Soft hint passed as ``joblib.Parallel(prefer=...)`` to parallelize
        the evaluation of the per-criterion sub-problems. One of
        ``"threads"``, ``"processes"``, or ``None`` for joblib's default
        behavior.

    n_jobs : int or None, default ``None``
        Number of parallel jobs used to evaluate the per-criterion
        sub-problems. ``None`` means sequential execution (joblib's
        default of a single job); ``-1`` uses all available processors.

    Notes
    -----
    Every concrete checker reads its notion of importance as either
    *necessity* or *sufficiency*, picked via the ``_invert_similarity``
    class attribute it must define:

    - **Necessity** (``_invert_similarity = True``): importance is
      ``1 - similarity`` to the reference. A criterion is important if
      the ranking *cannot do without it* -- removing it changes the
      ranking a lot. :class:`~skcriteria.importance.\
criteria_leave_one_out.CriteriaLeaveOneOutChecker` reads importance this
      way.
    - **Sufficiency** (``_invert_similarity = False``): importance is
      the similarity to the reference as is. A criterion is important if
      it *alone is enough* -- its sub-problem ranking already looks like
      the reference. :class:`~skcriteria.importance.criteria_keep_only_one.\
CriteriaKeepOnlyOneChecker` reads importance this way.

    Both orientations are normalized so that, regardless of checker or
    ``metric``, 0 always means "matters little" and 1 always means
    "matters a lot".

    Raises
    ------
    TypeError
        If ``dmaker`` doesn't implement the required ``evaluate()`` method.
    ValueError
        If ``metric`` is not ``"footrule"`` or ``"kendall"``.

    """

    _skcriteria_abstract_class = True
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
        """True if sub-problem rankings are allowed to be missing \
        alternatives with respect to the original decision matrix."""
        return self._allow_missing_alternatives

    @property
    def preferred_parallel_backend(self):
        """Backend used to parallelize the evaluation of the \
        per-criterion sub-problems."""
        return self._preferred_parallel_backend

    @property
    def n_jobs(self):
        """Number of parallel jobs used to evaluate the per-criterion \
        sub-problems."""
        return self._n_jobs

    # ABSTRACT ================================================================

    # `_invert_similarity` and `_prefix` are also required in every
    # concrete subclass (no default here on purpose -- forgetting to set
    # either should raise an AttributeError, not silently pick an
    # orientation or drop the sub-problem's `extra`). See the class
    # docstring's Notes section for the necessity/sufficiency distinction
    # `_invert_similarity` picks between; `_prefix` is both the key under
    # which `_patch_rank` nests each sub-problem's `extra` dict inside the
    # patched ranking's `extra_`, and the prefix each concrete checker
    # uses to name its sub-problem rankings as `"<prefix>(<criterion>)"`
    # (e.g. `"LOO"`, `"KOO"`) -- `_importance_score` reads the criterion
    # back out of `extra_[self._prefix]["criterion"]`, not by parsing the
    # ranking name.

    @abc.abstractmethod
    def _evaluate_subproblem(self, dm, criterion):
        """Build the sub-problem for ``criterion`` and evaluate \
        ``self.dmaker`` on it.

        This, together with ``_invert_similarity``, is all a concrete
        checker needs to define; it is what actually differentiates one
        checker from another (e.g. dropping ``criterion`` vs. keeping
        only ``criterion``).

        Defined as an instance method (instead of a free function) for
        simplicity; it is still safely parallelizable with ``joblib``
        since ``self`` only holds simple, picklable attributes.

        Parameters
        ----------
        dm : DecisionMatrix
            The (already weight-normalized) full decision matrix.
        criterion : str
            Name of the criterion the sub-problem is built around.

        Returns
        -------
        list of (rank_name, RankResult, extra)
            One entry per ranking produced for this ``criterion`` --
            normally just one, but a concrete checker may return more
            than one sub-problem ranking per criterion. For each entry:

            - ``rank_name`` (str): name to use for this ranking in the
              ``RanksComparator`` returned by :meth:`evaluate`.
            - ``RankResult``: ranking obtained by evaluating
              ``self.dmaker`` on the sub-problem.
            - ``extra`` (dict): what was changed to build this
              sub-problem, merged as-is into the resulting ranking's
              ``extra_`` under ``self._prefix``. Must contain at least a
              ``"criterion"`` key holding back the same ``criterion``
              received as input -- :meth:`_importance_score` reads it
              from there to re-index its result by criterion name. Any
              other key is up to the concrete checker.

        """
        raise NotImplementedError()

    # INTERNALS ===============================================================

    def _patch_rank(
        self,
        *,
        rank,
        full_alternatives,
        where,
        allow_missing_alternatives,
        extra,
    ):
        """Fill any alternative missing from ``rank`` (with respect to \
        ``full_alternatives``) with the worst rank + 1.

        Same convention used by
        ``skcriteria.ranksrev.rank_invariant_check.RankInvariantChecker``.

        ``allow_missing_alternatives`` is received as an explicit
        argument instead of read off ``self`` internally, so the
        (sequential, since it's cheap regardless -- the expensive part
        is ``dmaker.evaluate()``) patching loop in :meth:`evaluate`
        pulls it from ``self`` once and threads it through explicitly.

        ``extra``, if given, is merged as-is into the patched ranking's
        ``extra_`` -- this is how the ``extra`` dict returned by
        :meth:`_evaluate_subproblem` ends up visible on each sub-problem
        ranking.

        """
        method = str(rank.method)
        alternatives = rank.alternatives.copy()
        values = rank.values.copy()
        patched_extra = dict(rank.extra_.items())
        if extra is not None:
            patched_extra[self._prefix] = extra

        alts_diff = np.setxor1d(alternatives, full_alternatives)
        missing_alternatives = np.array([], dtype=full_alternatives.dtype)

        if len(alts_diff):
            if not allow_missing_alternatives:
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
            extra=patched_extra,
        )
        return patched_rank, missing_alternatives

    def _importance_score(self, named_ranks):
        """Importance of every criterion behind ``named_ranks`` vs \
        ``"reference"``.

        Returned as a ``pandas.Series`` indexed by criterion name, always
        bounded in ``[0, 1]``, where 0 always means "matters little" and 1
        always means "matters a lot" -- regardless of whether the concrete
        checker's notion of importance is necessity (LOO: the ranking
        changed a lot without this criterion) or sufficiency (OO: this
        criterion alone reproduces the ranking well).

        Builds the temporary ``RanksComparator`` needed to compute the
        pairwise metric once (reusing its own vectorized methods) and
        indexes the ``"reference"`` row out of it to get a similarity
        ``s``, normalized to ``[0, 1]`` regardless of ``metric``:
        ``footrule_similarity()`` is already bounded in ``[0, 1]``, so it
        is used as is; Kendall's tau is a correlation bounded in
        ``[-1, 1]`` instead, so it is rescaled with ``(1 + correlation) /
        2``. The subclass then picks the orientation via
        ``_invert_similarity``: necessity-style checkers report
        ``1 - s`` (important means *different* from the reference),
        sufficiency-style checkers report ``s`` as is (important means
        *similar* to the reference).
        """
        rcmp = RanksComparator(named_ranks, extra={})
        if self._metric == "footrule":
            similarity = rcmp.footrule_similarity(untied=self._untied)
            s = similarity.loc["reference"]
        else:
            correlation = rcmp.corr(method="kendall", untied=self._untied)
            s = (1.0 + correlation.loc["reference"]) / 2.0

        importance = (1.0 - s) if self._invert_similarity else s

        # comparing "reference" to itself isn't a real per-criterion
        # score, so drop it; what the caller cares about is the
        # criterion, not the ranking, so re-index by criterion name,
        # read back from each sub-problem ranking's `extra_`, where
        # `_evaluate_subproblem` echoed it under `self._prefix`
        criterion_by_name = {
            name: rank.extra_[self._prefix]["criterion"]
            for name, rank in named_ranks
            if name != "reference"
        }
        importance = importance.drop("reference")
        importance.index = [criterion_by_name[name] for name in importance.index]
        importance.name = "Importance"

        return importance

    # LOGIC ===================================================================

    def evaluate(self, dm):
        """Execute the importance test.

        Parameters
        ----------
        dm : DecisionMatrix
            The decision matrix to be evaluated. Must have at least 2
            criteria.

        Returns
        -------
        RanksComparator
            An object containing the reference ranking (named
            ``"reference"``) plus one ranking per criterion, named
            according to :meth:`_evaluate_subproblem`. The ``extra_``
            attribute contains:

            - ``metric``: the metric used (``"footrule"`` or ``"kendall"``).
            - ``importance``: a ``pandas.Series``, indexed by criterion
              name, always bounded in ``[0, 1]`` (0 means the sub-problem
              ranking is identical to the reference; 1 means the maximum
              possible difference for ``metric``). ``"reference"`` is not
              included, since it isn't a criterion.

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
                f"{self.get_method_name()} requires at least 2 criteria"
            )

        # normalize the weights of the full problem first, just in case,
        # so the reference ranking follows the same "weights sum to 1"
        # convention generally applied by concrete sub-problems
        dm = _WEIGHT_SCALER.transform(dm)

        full_alternatives = np.array(dm.alternatives)
        allow_missing_alternatives = self._allow_missing_alternatives

        # reference ranking, using all the criteria
        rank_full = self._dmaker.evaluate(dm)
        patched_full, _ = self._patch_rank(
            rank=rank_full,
            full_alternatives=full_alternatives,
            where="reference",
            allow_missing_alternatives=allow_missing_alternatives,
            extra=None,
        )

        names = ["reference"]
        results = [patched_full]

        # one sub-problem ranking per criterion, possibly in parallel; each
        # ranking already comes back named by `_evaluate_subproblem`
        with joblib.Parallel(
            n_jobs=self._n_jobs, prefer=self._preferred_parallel_backend
        ) as parallel:
            delayed_evaluation = joblib.delayed(self._evaluate_subproblem)
            sub_results = parallel(
                delayed_evaluation(dm, criterion) for criterion in criteria
            )

        # the patch itself is a cheap, self-contained function -- it is
        # applied sequentially, once the (possibly parallel) evaluations
        # come back, simply because there is nothing to parallelize: it
        # is O(n_alternatives) per ranking, dwarfed by dmaker.evaluate();
        # each criterion's call to `_evaluate_subproblem` returns a list
        # of (name, rank_sub, extra), so it's flattened here
        for sub_result in sub_results:
            for name, rank_sub, extra in sub_result:
                patched_sub, _ = self._patch_rank(
                    rank=rank_sub,
                    full_alternatives=full_alternatives,
                    where=f"ranking {name!r}",
                    allow_missing_alternatives=allow_missing_alternatives,
                    extra=extra,
                )

                names.append(name)
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
