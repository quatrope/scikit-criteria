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

from ._base_importance import CriteriaImportanceABC

# =============================================================================
# CLASS
# =============================================================================


class CriteriaOnlyOneChecker(CriteriaImportanceABC):
    r"""Only-one importance of each decision-matrix criterion.

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
    :class:`~skcriteria.importance.criteria_leave_one_out.CriteriaLeaveOneOutChecker`
    (see there for the details on how the similarity is computed and
    rescaled), reusing
    :meth:`skcriteria.cmp.RanksComparator.footrule_similarity` or
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

    def _evaluate_subproblem(self, dm, criterion):
        dm_sub = dm[criterion].replace(weights=[1])
        rank_sub = self._dmaker.evaluate(dm_sub)
        return f"OO-{criterion}", rank_sub
