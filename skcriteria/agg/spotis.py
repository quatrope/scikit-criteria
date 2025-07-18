#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""SPOTIS method."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank

# =============================================================================
# SPOTIS
# =============================================================================


def spotis(matrix, weights, bounds, isp):
    """Execute SPOTIS method."""
    min_bounds = bounds[:, 0]
    max_bounds = bounds[:, 1]

    # Calculate alternatives distances to ISP
    normalized_distance = np.abs((matrix - isp) / (max_bounds - min_bounds))

    # Scores by weighted sum
    scores = np.sum(normalized_distance * weights, axis=1)

    return rank.rank_values(scores), {"score": scores}


class SPOTIS(SKCDecisionMakerABC):
    r"""The Stable Preference Ordering Towards Ideal Solution (SPOTIS) method.

    The SPOTIS method is a multi-criteria decision analysis method that is
    exempt of rank reversal. The method is rank reversal free because the
    preference ordering established from the score matrix of the MCDM problem
    does not require relative comparisons between the alternatives, but only
    comparisons with respect to the ideal solution (ISP) chosen by the
    MCDM designer.

    References
    ----------
    :cite:p:`dezert2020spotis`
    """

    _skcriteria_parameters = []

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, weights, bounds, isp, **kwargs):
        extra = {"bounds": bounds, "isp": isp}

        rank, method_extra = spotis(matrix, weights, bounds, isp)
        extra.update(method_extra)

        return rank, extra

    def evaluate(self, dm, *, bounds=None, isp=None):
        """Validate the decision matrix and calculate a ranking.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.
        bounds: array-like, optional
            The bounds of the problem. If not provided, they will be calculated
            from the matrix.
        isp: array-like, optional
            The ideal solution point (ISP), if not provided, it will be
            calculated from the bounds.

        Raises
        ------
        ValueError:
            - If bounds are provided and the matrix has values out of the
            bounds.
            - If ISP is provided and the ISP has values out of the bounds
            (either given or calculated from the matrix).
            - If bounds or ISP have an invalid shape.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.

        """
        numpy_matrix = dm.matrix.to_numpy()
        if bounds is None:
            bounds = self._bounds_from_matrix(numpy_matrix)
        else:
            bounds = np.asarray(bounds)
            self._validate_bounds(bounds, numpy_matrix)

        if isp is None:
            isp = self._isp_from_bounds(bounds, dm.iobjectives.to_numpy())
        else:
            isp = np.asarray(isp)
            self._validate_isp(isp, bounds)

        return self._evaluate_dm(dm, bounds=bounds, isp=isp)

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "SPOTIS",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )

    def _bounds_from_matrix(self, matrix):
        """Calculate the bounds of the problem from the matrix."""
        min_bounds = np.min(matrix, axis=0).reshape(-1, 1)
        max_bounds = np.max(matrix, axis=0).reshape(-1, 1)
        return np.hstack((min_bounds, max_bounds))

    def _isp_from_bounds(self, bounds, objectives):
        """Calculate the reference or nominal Ideal Solution Point (ISP) \
        from the bounds and objectives."""
        row_indexs = np.arange(bounds.shape[0])
        col_indexs = [
            0 if obj == Objective.MIN.value else 1 for obj in objectives
        ]
        isp = bounds[row_indexs, col_indexs]

        return isp

    def _validate_bounds(self, bounds, matrix):
        if bounds.shape != (matrix.shape[1], 2):
            raise ValueError(
                f"Invalid shape for bounds. It must be (n_criteria, 2). \
                Got: {bounds.shape}."
            )

        min_bounds, max_bounds = bounds[:, 0], bounds[:, 1]

        within_bounds = (matrix >= min_bounds) & (matrix <= max_bounds)
        if not np.all(within_bounds):
            raise ValueError(
                "The matrix values must be within the provided bounds."
            )

    def _validate_isp(self, isp, bounds):
        if isp.shape[0] != bounds.shape[0]:
            raise ValueError(
                f"Invalid shape for Ideal Solution Point (ISP). It must \
                have the same number of criteria as the bounds. \
                Got: {isp.shape}."
            )

        min_bounds, max_bounds = bounds[:, 0], bounds[:, 1]
        if not np.all(isp >= min_bounds) or not np.all(isp <= max_bounds):
            raise ValueError(
                "The isp values must be within the provided bounds."
            )
