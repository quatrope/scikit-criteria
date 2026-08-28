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

from ._base_importance import CriteriaImportanceABC, _WEIGHT_SCALER

# =============================================================================
# CLASS
# =============================================================================


class CriteriaLeaveOneOutChecker(CriteriaImportanceABC):
    r"""Leave-one-out (LOO) importance of each decision-matrix criterion.

    For every criterion :math:`i` in the decision matrix, this checker
    builds a sub-problem :math:`N \setminus \{i\}` by dropping the criterion
    entirely and renormalizing the remaining weights so they sum to 1, then
    evaluates ``dmaker`` on that sub-problem. The resulting ranking is
    compared against the reference ranking (evaluated with all the
    criteria, :math:`N`) using a pairwise ranking-similarity metric, and the
    similarity between the two is reported as the "importance" of the
    removed criterion: the more the ranking changes when a criterion is
    removed, the more important that criterion was. This is a
    *necessity*-style reading of importance: a criterion is important if
    the ranking cannot do without it (see
    :class:`~skcriteria.importance._base_importance.CriteriaImportanceABC`
    for the necessity/sufficiency distinction shared by every checker).

    See
    :class:`~skcriteria.importance._base_importance.CriteriaImportanceABC`
    for the constructor parameters, shared by every criteria-importance
    checker.

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
    need to add up to any particular total). ``CriteriaKeepOnlyOneChecker`` is
    its complement at the opposite end of the coalition chain, measuring
    :math:`v(\{i\}) - v(\emptyset)`.

    The pairwise ranking similarity is always computed by reusing
    :meth:`skcriteria.cmp.RanksComparator.footrule_similarity` or
    :meth:`skcriteria.cmp.RanksComparator.corr`, instead of an ad hoc
    distance function, to stay consistent with the rest of the comparison
    ecosystem. The full pairwise matrix is computed once over the whole set
    of rankings (reference plus one leave-one-out ranking per criterion),
    and the per-criterion scores are obtained by taking the complement of
    the ``"reference"`` row of that single matrix. ``footrule_similarity``
    is already bounded in :math:`[0, 1]`, so its complement is used as is;
    Kendall's tau is a correlation bounded in :math:`[-1, 1]`, so its
    complement is halved instead, keeping the importance score in
    :math:`[0, 1]` regardless of ``metric``.

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
    >>> print(result.extra_["importance"])
    C0    0.25
    C1    0.25
    C2    0.25
    Name: Importance, dtype: float64

    """

    #: necessity: important means the ranking changed a lot without it.
    _invert_similarity = True

    #: key under which this checker's per-criterion `extra` dict is
    #: nested inside each sub-problem ranking's `extra_`, and the prefix
    #: used to name each sub-problem ranking (e.g. ``"LOO(C0)"``).
    _prefix = "LOO"

    def _evaluate_subproblem(self, dm, criterion, reference):
        """Evaluate ``dmaker`` with ``criterion`` dropped."""
        keep = [c for c in dm.criteria if c != criterion]
        dm_sub = _WEIGHT_SCALER.transform(dm[keep])

        rank_sub_name = f"{self._prefix}({criterion})"
        rank_sub = self._dmaker.evaluate(dm_sub)
        extra_sub = {"criterion": criterion}

        return [(rank_sub_name, rank_sub, extra_sub)]
