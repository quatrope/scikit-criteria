#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""One-at-a-time (OAT) criteria importance checker.

This module measures the importance of each criterion of a decision problem
by perturbing its weight up and down by a fixed fraction, renormalizing the
remaining weights so the whole vector still sums to 1, and comparing the
ranking obtained under each perturbed weight vector against a reference
ranking computed with the original weights.

Unlike leave-one-out or only-one, the criterion is never removed from the
problem: OAT asks "how much does the ranking move if I nudge this single
weight?", not "what happens if this criterion is absent?". This is the most
common form of weight-sensitivity analysis reported in the MCDA literature
(see :cite:`wieckowski2023`).

"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ._base_importance import CriteriaImportanceABC

# =============================================================================
# CLASS
# =============================================================================


class CriteriaOneAtATimeChecker(CriteriaImportanceABC):
    r"""One-at-a-time (OAT) importance of each decision-matrix criterion.

    For every criterion :math:`i` in the decision matrix, this checker
    perturbs its weight :math:`w_i` by a fixed relative amount ``delta``,
    both increasing and decreasing it, renormalizing the remaining weights
    :math:`w_j,\ j \neq i` so the whole vector still sums to 1 while
    preserving their relative proportions, and evaluates ``dmaker`` on
    each of the two perturbed sub-problems. Both perturbed rankings are
    scored against the reference ranking (evaluated once with the
    original weights) using the same pairwise ranking-similarity metric
    as every other checker in this module, and only the **worst case**
    (the direction that moves the ranking furthest from the reference) is
    kept -- the more the ranking *can* move when its weight is nudged in
    either direction, the more important that criterion is considered to
    be.

    This is a *necessity*-style reading of importance, like
    :class:`~skcriteria.importance.criteria_leave_one_out.\
CriteriaLeaveOneOutChecker`, but the perturbation here is local (a single
    weight moved by ``delta``) rather than total (the criterion dropped
    entirely). It is the checker that best matches how weight-sensitivity
    analysis is most commonly practiced in the applied MCDA literature.
    There is deliberately no parameter to restrict the check to a single
    direction: a sensitivity analysis that only looked at one direction
    would silently under-report a criterion whose ranking is fragile only
    on the other side, so both directions are always evaluated and the
    worse one is what gets reported.

    Parameters
    ----------
    dmaker : object
        Decision maker instance implementing ``evaluate(dm)``.

    delta : float, default 0.2
        Relative perturbation applied to each weight, e.g. ``0.2`` moves
        :math:`w_i` to :math:`w_i \cdot (1 \pm 0.2)` before renormalizing.
        Must be in :math:`(0, 1)` (a delta of 1 would zero out or double
        the original weight before renormalization).

    metric, untied, allow_missing_alternatives, preferred_parallel_backend,\
 n_jobs
        Inherited as-is from
        :class:`~skcriteria.importance._base_importance.CriteriaImportanceABC`;
        see that class for the full description of each.

    Notes
    -----
    OAT is a *local*, single-point sample of the same coalition space that
    Shapley values traverse exhaustively: it only ever compares the full
    coalition :math:`N` against a small perturbation of itself, and never
    considers what happens when several weights move together, or when a
    criterion's weight is evaluated from a very different starting
    coalition.

    Only the worse of the two perturbed rankings is kept for a given
    criterion, chosen with
    :meth:`~skcriteria.importance._base_importance.CriteriaImportanceABC.\
_similarity_to_reference` against the ``reference`` ranking handed to
    :meth:`_evaluate_subproblem` -- so the reference is evaluated once per
    :meth:`evaluate` call, not once per direction per criterion. It shows
    up as a single entry per criterion in the
    :class:`~skcriteria.cmp.RanksComparator` returned by :meth:`evaluate`,
    named ``"OAT(<criterion>+<delta>)"`` or ``"OAT(<criterion>-<delta>)"``
    depending on which direction lost.

    """

    #: necessity: important means the ranking changed a lot under a small
    #: perturbation of this single weight.
    _invert_similarity = True

    #: key under which this checker's per-criterion `extra` dict is
    #: nested inside each sub-problem ranking's `extra_`, and the prefix
    #: used to name each sub-problem ranking (e.g. ``"OAT(C0+0.2)"``).
    _prefix = "OAT"

    _skcriteria_parameters = CriteriaImportanceABC._skcriteria_parameters + [
        "delta",
    ]

    def __init__(self, dmaker, *, delta=0.2, **kwargs):
        super().__init__(dmaker, **kwargs)

        delta = float(delta)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"'delta' must be in (0, 1). Found {delta!r}")
        self._delta = delta

    # PROPERTIES ==============================================================

    @property
    def delta(self):
        """Relative perturbation applied to each weight."""
        return self._delta

    # INTERNALS ===============================================================

    def _perturbed_weights(self, dm, criterion, sign):
        """Return a full weight vector with ``criterion`` moved by \
        ``sign * delta`` and the rest renormalized proportionally."""
        weights = dm.weights.copy()
        w_i = weights[criterion]

        new_w_i = w_i * (1.0 + sign * self._delta)

        others = weights.drop(criterion)
        # keep the relative proportions of every other weight, but rescale
        # them so the whole vector sums back to 1
        remaining_mass = 1.0 - new_w_i
        others_rescaled = others * (remaining_mass / others.sum())

        new_weights = others_rescaled.copy()
        new_weights[criterion] = new_w_i
        return new_weights.reindex(dm.criteria)

    def _evaluate_subproblem(self, dm, criterion, reference):
        """Evaluate ``dmaker`` with ``criterion``'s weight perturbed by \
        both ``+delta`` and ``-delta``, keeping only the worse of the two."""
        # `reference` was already patched by `evaluate()` to include every
        # alternative in `dm`; a candidate must match that same set before
        # `_similarity_to_reference` can compare it, so patch it here too
        # (harmless if `dmaker` never drops alternatives to begin with --
        # `evaluate()` re-patching the returned ranking afterwards is then
        # a no-op).
        full_alternatives = np.array(dm.alternatives)

        candidates = []
        for sign in (+1, -1):
            weights = self._perturbed_weights(dm, criterion, sign)
            rank_sub = self._dmaker.evaluate(dm.replace(weights=weights.values))
            patched_sub, _ = self._patch_rank(
                rank=rank_sub,
                full_alternatives=full_alternatives,
                where=f"sub-problem for {criterion!r}",
                allow_missing_alternatives=self._allow_missing_alternatives,
                extra=None,
            )
            similarity = self._similarity_to_reference(reference, patched_sub)
            candidates.append((sign, patched_sub, similarity))

        # worst case = lowest similarity to the reference (this checker is
        # necessity-style: importance grows the further the ranking moves)
        sign, rank_sub, _ = min(candidates, key=lambda c: c[2])

        tag = {1: "+", -1: "-"}[sign]
        rank_sub_name = f"{self._prefix}({criterion}{tag}{self._delta})"
        extra_sub = {"criterion": criterion, "direction": tag}

        return [(rank_sub_name, rank_sub, extra_sub)]
