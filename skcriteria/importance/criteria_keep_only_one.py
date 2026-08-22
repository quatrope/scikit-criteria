#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Keep-only-one criteria importance checker.

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

from ._base_importance import CriteriaImportanceABC

# =============================================================================
# CLASS
# =============================================================================


class CriteriaKeepOnlyOneChecker(CriteriaImportanceABC):
    r"""Keep-only-one importance of each decision-matrix criterion.

    For every criterion :math:`i` in the decision matrix, this checker
    builds a sub-problem :math:`\{i\}` keeping *only* that criterion
    (weight set to 1), evaluates ``dmaker`` on that sub-problem, and
    compares the resulting ranking against the reference ranking
    (evaluated with all the criteria, :math:`N`) using a pairwise
    ranking-similarity metric.

    See
    :class:`~skcriteria.importance._base_importance.CriteriaImportanceABC`
    for the constructor parameters, shared by every criteria-importance
    checker.

    Notes
    -----
    This checker uses the same pairwise-comparison machinery as
    :class:`~skcriteria.importance.criteria_leave_one_out.CriteriaLeaveOneOutChecker`,
    reusing :meth:`skcriteria.cmp.RanksComparator.footrule_similarity` or
    :meth:`skcriteria.cmp.RanksComparator.corr` instead of an ad hoc
    distance function, but on the opposite sub-problem: instead of
    dropping one criterion from the full set :math:`N`, it keeps *only*
    one criterion from the empty set :math:`\emptyset`.

    Unlike ``CriteriaLeaveOneOutChecker`` (which reports *how different*
    the ranking becomes), this checker reports the *similarity* between
    the single-criterion ranking and the reference ranking directly: a
    criterion whose ranking alone already looks like the reference gets
    high importance (it is *sufficient* on its own), while one that alone
    produces a very different ranking gets low importance. This is the
    opposite orientation of ``CriteriaLeaveOneOutChecker``'s *necessity*
    reading, but both are normalized so that, in either checker, 0 always
    means "matters little" and 1 always means "matters a lot".

    """

    #: sufficiency: important means this criterion alone already looks
    #: like the reference.
    _invert_similarity = False

    #: key under which this checker's per-criterion `extra` dict is
    #: nested inside each sub-problem ranking's `extra_`.
    _extra_key = "koo"

    def _evaluate_subproblem(self, dm, criterion):
        """Evaluate ``dmaker`` with only ``criterion`` kept."""
        dm_sub = dm[criterion].replace(weights=[1])
        rank_sub = self._dmaker.evaluate(dm_sub)
        extra = {"criteria": {"kept": criterion}}
        return f"OO-{criterion}", rank_sub, extra
