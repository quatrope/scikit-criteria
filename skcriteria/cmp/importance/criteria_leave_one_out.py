#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Leave-one-out (LOO) criteria importance checker.

This module measures the importance of each criterion of a decision problem
by completely removing it (as opposed to perturbing it) from the decision
matrix, renormalizing the remaining weights, and comparing the ranking
obtained without that criterion against a reference ranking computed with
all the criteria.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ...agg import RankResult
from ...cmp import RanksComparator
from ...core import SKCMethodABC
from ...utils import Bunch, unique_names

# =============================================================================
# CONSTANTS
# =============================================================================

_VALID_METRICS = frozenset(["footrule", "kendall"])


# =============================================================================
# CLASS
# =============================================================================


class CriteriaLeaveOneOutChecker(SKCMethodABC):
    r"""Leave-one-out (LOO) importance of every criterion in a decision \
    problem.

    For every criterion :math:`i` in the decision matrix, this checker
    builds a sub-problem :math:`N \setminus \{i\}` by dropping the criterion
    entirely and renormalizing the remaining weights so they sum to 1, then
    evaluates ``dmaker`` on that sub-problem. The resulting ranking is
    compared against the reference ranking (evaluated with all the
    criteria, :math:`N`) using a pairwise ranking-similarity metric, and the
    similarity between the two is reported as the "importance" of the
    removed criterion: the more the ranking changes when a criterion is
    removed, the more important that criterion was.

    Parameters
    ----------
    dmaker : object
        Decision maker instance that must implement an ``evaluate(dm)``
        method returning a ``skcriteria.agg.RankResult``. This represents
        the MCDA method or pipeline whose sensitivity to each criterion is
        being measured.

    metric : str, default ``"footrule"``
        Ranking-similarity metric used to compare a leave-one-out ranking
        against the reference ranking. One of:

        - ``"footrule"``: normalized Spearman footrule similarity, computed
          with :meth:`skcriteria.cmp.RanksComparator.footrule_similarity`.
        - ``"kendall"``: Kendall tau rank correlation, computed with
          :meth:`skcriteria.cmp.RanksComparator.corr` using
          ``method="kendall"``.

    allow_missing_alternatives : bool, default ``True``
        ``dmaker`` can somehow return rankings with fewer alternatives than
        the original decision matrix (using a pipeline that implements a
        filter, for example). If ``True``, any alternative missing from a
        ranking (the reference or any leave-one-out ranking) is added back
        with a value equal to the worst rank obtained in that ranking plus
        1. If ``False``, a ranking missing an alternative raises a
        ``ValueError``.

    Raises
    ------
    TypeError
        If ``dmaker`` doesn't implement the required ``evaluate()`` method.
    ValueError
        If ``metric`` is not ``"footrule"`` or ``"kendall"``.

    Notes
    -----
    This checker is closely related to the Shapley value of cooperative
    game theory. If :math:`v(S)` is the "value" (here, the ranking
    obtained) of using the subset of criteria :math:`S`, the Shapley value
    of criterion :math:`i` averages the marginal contribution
    :math:`v(S \cup \{i\}) - v(S)` over every possible coalition
    :math:`S \subseteq N \setminus \{i\}` and every order in which criteria
    could be added. The importance score computed here is exactly the
    marginal contribution :math:`v(N) - v(N \setminus \{i\})`, evaluated at
    a single point of that chain: the coalition of *all* the other
    criteria. In other words, LOO is a Shapley marginal contribution
    evaluated on a single input order, not averaged over all of them, so it
    carries no guarantee of additivity or efficiency (the LOO scores do not
    need to add up to any particular total). ``CriteriaOnlyOneChecker`` is
    its complement at the opposite end of the coalition chain, measuring
    :math:`v(\{i\}) - v(\emptyset)`.

    The pairwise ranking similarity is always computed by reusing
    :meth:`skcriteria.cmp.RanksComparator.footrule_similarity` or
    :meth:`skcriteria.cmp.RanksComparator.corr`, instead of an ad hoc
    distance function, to stay consistent with the rest of the comparison
    ecosystem. Both are computed once over the whole set of rankings
    (reference plus one leave-one-out ranking per criterion), and the
    per-criterion scores are obtained by indexing that single matrix.

    Examples
    --------
    >>> from skcriteria.agg import simple
    >>> from skcriteria import mkdm
    >>>
    >>> dm = mkdm(
    ...     matrix=[[1, 2, 3], [4, 5, 6], [3, 1, 2]],
    ...     objectives=[max, max, max],
    ...     alternatives=["A", "B", "C"],
    ...     criteria=["C0", "C1", "C2"],
    ... )
    >>>
    >>> dmaker = simple.WeightedSumModel()
    >>> checker = CriteriaLeaveOneOutChecker(dmaker)
    >>> result = checker.evaluate(dm)
    >>>
    >>> print(result.extra_["importance_scores"])
    >>> print(result.extra_["similarity_matrix"])

    """

    _skcriteria_dm_type = "sensitivity_importance"
    _skcriteria_parameters = [
        "dmaker",
        "metric",
        "allow_missing_alternatives",
    ]

    def __init__(
        self, dmaker, *, metric="footrule", allow_missing_alternatives=True
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

        self._allow_missing_alternatives = bool(allow_missing_alternatives)

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
    def allow_missing_alternatives(self):
        """True if leave-one-out rankings are allowed to be missing \
        alternatives with respect to the original decision matrix."""
        return self._allow_missing_alternatives

    # INTERNALS ===============================================================

    def _drop_criterion(self, dm, criterion):
        """Build the decision matrix without ``criterion``, with the \
        remaining weights renormalized to sum 1."""
        keep = [c for c in dm.criteria if c != criterion]
        dm_dropped = dm[keep]
        weights = dm_dropped.weights
        renormalized_weights = (weights / weights.sum()).to_numpy()
        return dm_dropped.replace(weights=renormalized_weights)

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

            # add missing alternatives with the worst ranking + 1
            fill_values = np.full_like(alts_diff, rank.rank_.max() + 1)
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

    def _similarity_matrix(self, rcmp):
        """Compute the pairwise ranking-similarity matrix for ``rcmp``, \
        reusing ``RanksComparator``'s own vectorized methods."""
        if self._metric == "footrule":
            return rcmp.footrule_similarity()
        return rcmp.corr(method="kendall")

    # LOGIC ===================================================================

    def evaluate(self, dm):
        """Execute the leave-one-out importance test.

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
            ``"LOO(-{criterion})"``), obtained by evaluating ``dmaker``
            without that criterion. The ``extra_`` attribute contains:

            - ``loo_check``: a mapping from criterion name to a
              :class:`~skcriteria.utils.Bunch` with the leave-one-out
              ``rank``, its importance score (``delta``), and any
              ``missing_alternatives`` that had to be patched.
            - ``metric``: the metric used (``"footrule"`` or ``"kendall"``).
            - ``similarity_matrix``: the full pairwise similarity/
              correlation matrix (same one used to obtain every
              ``delta``), for traceability.
            - ``importance_scores``: a ``{criterion: delta}`` dict with the
              headline result of this checker. As explained in the class
              Notes, this is **not** a Shapley value: it is
              :math:`v(N) - v(N \\setminus \\{i\\})` evaluated at a single
              coalition order, without any additivity/efficiency
              guarantee.

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
                "CriteriaLeaveOneOutChecker requires at least 2 criteria"
            )

        full_alternatives = np.array(dm.alternatives)

        # reference ranking, using all the criteria
        rank_full = self._dmaker.evaluate(dm)
        patched_full, _ = self._patch_missing_alternatives(
            rank=rank_full,
            full_alternatives=full_alternatives,
            where="the reference ranking",
        )

        names = ["reference"]
        results = [patched_full]

        # one leave-one-out ranking per criterion
        loo_info = {}
        for criterion in criteria:
            dm_sub = self._drop_criterion(dm, criterion)
            rank_sub = self._dmaker.evaluate(dm_sub)

            patched_sub, missing = self._patch_missing_alternatives(
                rank=rank_sub,
                full_alternatives=full_alternatives,
                where=(
                    "the leave-one-out ranking for criterion "
                    f"{criterion!r}"
                ),
            )

            names.append(f"LOO(-{criterion})")
            results.append(patched_sub)

            loo_info[criterion] = {
                "rank": patched_sub,
                "missing_alternatives": missing,
            }

        # build the single RanksComparator that holds every ranking, and
        # compute the pairwise similarity matrix over it a single time
        named_ranks = unique_names(names=names, elements=results)
        rcmp = RanksComparator(named_ranks, extra={})
        sim_matrix = self._similarity_matrix(rcmp)

        # index the point-wise (loo, reference) pair already computed above,
        # instead of recomputing anything ranking by ranking
        importance_scores = {}
        for criterion in criteria:
            delta = sim_matrix.loc[f"LOO(-{criterion})", "reference"]
            importance_scores[criterion] = delta
            loo_info[criterion]["delta"] = delta

        loo_check = Bunch(
            "loo_check",
            {
                criterion: Bunch(f"loo_check[{criterion}]", info)
                for criterion, info in loo_info.items()
            },
        )

        extra = {
            "loo_check": loo_check,
            "metric": self._metric,
            "similarity_matrix": sim_matrix,
            "importance_scores": importance_scores,
        }

        return RanksComparator(named_ranks, extra=extra)
