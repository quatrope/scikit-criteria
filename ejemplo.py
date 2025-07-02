from skcriteria.agg import SKCDecisionMakerABC, RankResult
import numpy as np
import skcriteria as skc


class ExamplePrimaryDM(SKCDecisionMakerABC):
    """Decision maker que devuelve [1,1,2,2,3] hardcodeado."""

    _skcriteria_parameters = []

    def _evaluate_data(self, **kwargs):
        return [1, 1, 2, 2, 3], {}

    def _make_result(self, alternatives, values, extra):
        return RankResult(
            method="ExamplePrimary",
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


alternatives = ["A", "B", "C", "D", "E"]

# 3 criterios: Cost (min), Quality (max), Speed (max)
criteria = ["Cost", "Quality", "Speed"]

# Matriz de valores (5 alternativas x 3 criterios)
matrix = np.array(
    [
        [100, 8.5, 7.2],  # A
        [120, 8.5, 6.8],  # B - similar quality a A
        [150, 9.2, 8.1],  # C
        [140, 9.2, 7.9],  # D - similar quality a C
        [200, 9.8, 9.0],  # E - mejor en todo pero más caro
    ]
)

# Objetivos: minimizar costo, maximizar calidad y velocidad
objectives = [max, max, max]

# Pesos iguales para todos los criterios
weights = [1 / 3, 1 / 3, 1 / 3]

# Crear la matriz de decisión
dm = skc.mkdm(
    matrix=matrix,
    objectives=objectives,
    alternatives=alternatives,
    criteria=criteria,
    weights=weights,
)


dmaker = ExamplePrimaryDM()
rank = dmaker.evaluate(dm)

from skcriteria import tiebreaker
from skcriteria.agg import simple


tb = tiebreaker.TieBreaker(
    dmaker=dmaker, untier=simple.WeightedSumModel(), force=True
)

tb.evaluate(dm)