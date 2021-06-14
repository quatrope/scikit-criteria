# el tipo de datos Data de skcriteria, debe contener:
# - La matriz de alternativas (mtx).
# - El sentido de los criterios (sense)
# - Pesos (weights)
# - nombre de los atributos (anames)
# - nombre de los criterios (cnames)

# Diseño:
# Con

import itertools as it

import attr
import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

MIN = -1
"""Int: Minimization criteria"""

MIN_ALIASES = [MIN, min, np.min, np.nanmin, np.amin, "min", "minimize"]
"""Another ways to name the minimization criteria."""

MAX = 1
"""Int: Maximization criteria"""

MAX_ALIASES = [MAX, max, np.max, np.nanmax, np.amax, "max", "maximize"]
"""Another way to name the maximization criteria."""


CRITERIA_STR = {MIN: "min", MAX: "max"}

CRITERIA_ALIASES = dict(
    it.chain(
        {MIN: MIN, MAX: MAX}.items(),
        dict.fromkeys(MIN_ALIASES, MIN).items(),
        dict.fromkeys(MAX_ALIASES, MAX).items(),
    )
)

CRITERIA_COLUMN = "criteria"

WEIGHTS_COLUMN = "weights"

# =============================================================================
# CONVERTE
# =============================================================================


def ascriteria(criteria):
    """Validate and convert a criteria array

    Check if the iterable only contains MIN (or any alias) and MAX
    (or any alias) values. And also always returns an ndarray representation
    of the iterable.

    Parameters
    ----------
    criteria : Array-like
        Iterable containing all the values to be validated by the function.

    Returns
    -------
    numpy.ndarray :
        Criteria array as intergers (-1 for minimize, 1 for maximize).

    """
    pcriteria = np.empty(len(criteria), dtype=int)
    for idx, crit in enumerate(criteria):
        crit = crit.lower() if isinstance(crit, str) else crit
        crit = CRITERIA_ALIASES.get(crit, None)
        if crit is None:
            raise ValueError(
                "Criteria Array only accept minimize or maximize Values. "
                f"Found {crit}"
            )
        pcriteria[idx] = crit
    return pcriteria


# =============================================================================
# DATA CLASS
# =============================================================================


@attr.s(frozen=True, repr=False, cmp=False)
class DecisionMatrix:

    data_df: pd.DataFrame = attr.ib()
    cw_df: pd.DataFrame = attr.ib()

    @property
    def anames(self):
        return self.data_df.index.to_numpy()

    @property
    def cnames(self):
        return self.data_df.columns.to_numpy()

    @property
    def mtx(self):
        return self.data_df.to_numpy()

    @property
    def weights(self):
        return self.cw_df.weights.to_numpy()

    @property
    def criteria(self):
        return self.cw_df.criteria.to_numpy()

    @property
    def dtypes(self):
        return self.data_df.dtypes

    def copy(self, deep=True):
        return DecisionMatrix(
            data_df=self.data_df.copy(deep=deep),
            cw_df=self.cw_df.copy(deep=deep),
        )

    def __eq__(self, other):
        return (
            isinstance(other, DecisionMatrix)
            and self.data_df.equals(other.data_df)
            and self.cw_df.equals(other.cw_df)
        )

    def __ne__(self, other):
        return not self == other


# =============================================================================
# factory
# =============================================================================


def mkdm(mtx, criteria, weights=None, anames=None, cnames=None):
    a_number, c_number = np.shape(mtx)

    anames = (
        np.array([f"A{idx}" for idx in range(a_number)])
        if anames is None
        else anames
    )
    cnames = (
        np.array([f"C{idx}" for idx in range(c_number)])
        if cnames is None
        else cnames
    )

    weights = np.ones(c_number) if weights is None else weights
    criteria = ascriteria(criteria)

    data_df = pd.DataFrame(mtx, index=anames, columns=cnames)
    cw_df = pd.DataFrame(
        {CRITERIA_COLUMN: criteria, WEIGHTS_COLUMN: weights}, index=cnames
    )

    return DecisionMatrix(data_df=data_df, cw_df=cw_df)
