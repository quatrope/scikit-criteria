from ..utils import hidden

with hidden():
    import numpy as np

    from ._agg_base import RankResult, SKCDecisionMakerABC
    from ..core import Objective
    from ..utils import doc_inherit, rank


def incrising_value_function(reference_point, values, alpha, lambd):
    gains = values > reference_point
    losses = ~gains

    result = np.empty_like(values, dtype=float)
    result[gains] = (values[gains] - reference_point) ** alpha
    result[losses] = -lambd * ((reference_point - values[losses]) ** alpha)

    return result


def decreasing_value_function(reference_point, values, alpha, lambd):
    gains = values < reference_point
    losses = ~gains

    result = np.empty_like(values, dtype=float)
    result[gains] = (reference_point - values[gains]) ** alpha
    result[losses] = -lambd * ((values[losses] - reference_point) ** alpha)

    return result


def ervd(matrix, objectives, weights, reference_points, alpha, lambd):
    """
    Execute ERVD without any validation.
    """

    for j in range(matrix.shape[1]):
        if objectives[j] == Objective.MAX.value:
            matrix[:, j] = incrising_value_function(
                reference_points[j], matrix[:, j], alpha, lambd
            )
        else:
            matrix[:, j] = decreasing_value_function(
                reference_points[j], matrix[:, j], alpha, lambd
            )

    # create the ideal and the anti ideal arrays
    ideal = np.max(matrix, axis=0)
    anti_ideal = np.min(matrix, axis=0)

    # calculate distances
    s_plus = np.sum(weights * np.abs(matrix - ideal), axis=1)
    s_minus = np.sum(weights * np.abs(matrix - anti_ideal), axis=1)

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
    ERVD (election based on relative value distances).
    """

    _skcriteria_parameters = []

    def __init__(self, reference_points, lambd=2.25, alpha=0.88):
        self.lambd = lambd
        self.alpha = alpha
        self.reference_points = reference_points

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            "ERVD", alternatives=alternatives, values=values, extra=extra
        )

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, matrix, objectives, weights, **kwargs):  # type: ignore
        rank, similarity, ideal, anti_ideal, s_plus, s_minus = ervd(
            matrix,
            objectives,
            weights,
            self.reference_points,
            self.alpha,
            self.lambd,
        )

        return rank, {
            "similarity": similarity,
            "ideal": ideal,
            "anti_ideal": anti_ideal,
            "s_plus": s_plus,
            "s_minus": s_minus,
        }
