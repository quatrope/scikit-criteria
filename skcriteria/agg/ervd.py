#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of ERVD method."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import functools

    import numpy as np
    from scipy.spatial import distance

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank

# =============================================================================
# ERVD
# =============================================================================


def _value_function(matrix, reference_points, alpha, lambd, objectives):
    """Value function for ERVD."""
    delta = matrix - reference_points  # Calculate the difference only one time

    maximize_mask = np.broadcast_to(objectives == Objective.MAX, matrix.shape)

    result = np.empty_like(matrix, dtype=float)

    # MAX objectives
    # masks
    gains_min = (delta < 0) & ~maximize_mask
    losses_min = ~gains_min & ~maximize_mask
    gains_max = (delta > 0) & maximize_mask
    losses_max = ~gains_max & maximize_mask

    # apply the value function
    result[gains_max] = delta[gains_max] ** alpha
    result[losses_max] = -lambd * ((-delta[losses_max]) ** alpha)

    # MIN objectives
    result[gains_min] = (-delta[gains_min]) ** alpha
    result[losses_min] = -lambd * (delta[losses_min] ** alpha)

    return result


w_minkowski = functools.partial(distance.minkowski, p=1)


def ervd(
    matrix,
    objectives,
    weights,
    reference_points,
    alpha,
    lambd,
    metric,
    w_metric,
    **kwargs,
):
    """Execute ERVD without any validation."""
    # apply the value function based on the maximize_mask
    value_matrix = _value_function(
        matrix, reference_points, alpha, lambd, objectives
    )

    # create the ideal and the anti ideal arrays
    ideal = np.max(value_matrix, axis=0)
    anti_ideal = np.min(value_matrix, axis=0)

    # calculate distances
    weight_or_none = weights if w_metric else None
    s_plus = distance.cdist(
        value_matrix, [ideal], metric=metric, w=weight_or_none, **kwargs
    ).flatten()
    s_minus = distance.cdist(
        value_matrix, [anti_ideal], metric=metric, w=weight_or_none, **kwargs
    ).flatten()

    # relative closeness
    similarity = s_minus / (s_plus + s_minus)

    return (
        rank.rank_values(similarity, reverse=True),
        similarity,
        ideal,
        anti_ideal,
        s_plus,
        s_minus,
    )


class ERVD(SKCDecisionMakerABC):
    """
    Election based on Relative Value Distances (ERVD) decision-making method.

    This method integrates an s-shape value function, departing from the
    traditional expected utility function, to more accurately capture
    risk-averse and risk-seeking behaviors. ERVD builds upon the foundational
    principles of the TOPSIS method, extending its capabilities by
    incorporating concepts from prospect theory to refine the assessment of
    alternatives based on their relative distances from ideal and anti-ideal
    solutions.

    Parameters
    ----------
    lambda_value: float, default=2.25
        Represents the attenuation factor of the losse.

    alpha_value: float, default=0.88
        Diminishing sensitivity parameters.

    metric: str or callable, default='minkowski'
        The distance metric to be used for calculating distances between
        alternatives and ideal/anti-ideal points. It can be a string
        representing a metric name from `scipy.spatial.distance` or a custom
        callable function that computes distances.

    w_metric: bool, default=True
        Whether to use weights in the distance metric calculation. If True,
        the weights will be applied to the alternatives when calculating
        distances. If False, the distances will be calculated without weights.

    References
    ----------
    :cite:p:`shyur2015multiple`
    """

    _skcriteria_parameters = [
        "lambda_value",
        "alpha_value",
        "metric",
        "w_metric",
    ]

    def __init__(
        self,
        *,
        lambda_value=2.25,
        alpha_value=0.88,
        metric=w_minkowski,
        w_metric=True,
    ):
        if not callable(metric) and metric not in distance._METRICS_NAMES:
            metrics = ", ".join(f"'{m}'" for m in distance._METRICS_NAMES)
            raise ValueError(
                f"Invalid metric '{metric}'. Plese choose from: {metrics}"
            )

        self._metric = metric
        self._lambd = lambda_value
        self._alpha = alpha_value
        self._w_metric = w_metric

    @property
    def alpha_value(self):
        """Diminishing sensitivity parameter."""
        return self._alpha

    @property
    def lambda_value(self):
        """Attenuation factor of the losses."""
        return self._lambd

    @property
    def metric(self):
        """Which distance metric will be used."""
        return self._metric

    @property
    def w_metric(self):
        """Whether to use weights in the metric."""
        return self._w_metric

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ERVD", alternatives=alternatives, values=values, extra=extra
        )

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(
        self, matrix, objectives, weights, reference_points, **kwargs
    ):
        rank, similarity, ideal, anti_ideal, s_plus, s_minus = ervd(
            matrix,
            objectives,
            weights,
            reference_points,
            self.alpha_value,
            self.lambda_value,
            self.metric,
            self.w_metric,
        )

        return rank, {
            "similarity": similarity,
            "ideal": ideal,
            "anti_ideal": anti_ideal,
            "s_plus": s_plus,
            "s_minus": s_minus,
        }

    def _validate_reference_points(self, reference_points, matrix):
        if reference_points is None:
            raise ValueError(
                "Reference points must be provided for ERVD evaluation."
            )
        if len(reference_points) != matrix.shape[1]:
            raise ValueError(
                "Reference points must match the number of criteria in "
                "the decision matrix."
            )

    def evaluate(self, dm, *, reference_points=None):
        """Validate the dm and calculate and evaluate the alternatives.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.
        reference_points: array-like, optional
            Reference points for each criterion.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.
        """
        data = dm.to_dict()

        self._validate_reference_points(reference_points, data["matrix"])

        result_data, extra = self._evaluate_data(
            **data, reference_points=reference_points
        )

        alternatives = data["alternatives"]
        result = self._make_result(
            alternatives=alternatives, values=result_data, extra=extra
        )

        return result
