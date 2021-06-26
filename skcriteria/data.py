# el tipo de datos Data de skcriteria, debe contener:
# - La matriz de alternativas (mtx).
# - El sentido de los criterios (sense)
# - Pesos (weights)
# - nombre de los atributos (anames)
# - nombre de los criterios (cnames)

# Diseño:
# Con

import enum
import itertools as it

import attr
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


# =============================================================================
# CONSTANTS
# =============================================================================
class Objective(enum.Enum):
    MIN = -1
    MAX = 1

    # INTERNALS ===============================================================

    _MIN_STR = "\u25bc"
    _MAX_STR = "\u25b2"

    #: Another way to name the maximization criteria.
    _MAX_ALIASES = frozenset(
        [
            MAX,
            _MAX_STR,
            max,
            np.max,
            np.nanmax,
            np.amax,
            "max",
            "maximize",
            "+",
            ">",
        ]
    )

    #: Another ways to name the minimization criteria.
    _MIN_ALIASES = frozenset(
        [
            MIN,
            _MIN_STR,
            min,
            np.min,
            np.nanmin,
            np.amin,
            "min",
            "minimize",
            "<",
            "-",
        ]
    )

    # CUSTOM CONSTRUCTOR ======================================================

    @classmethod
    def construct_from_alias(cls, alias):
        if isinstance(alias, cls):
            return alias
        if isinstance(alias, str):
            alias = alias.lower()
        if alias in cls._MAX_ALIASES.value:
            return cls.MAX
        if alias in cls._MIN_ALIASES.value:
            return cls.MIN
        raise ValueError(f"Invalid criteria objective {alias}")

    # METHODS =================================================================

    def __str__(self):
        return self.name

    def to_string(self):
        if self.value in Objective._MIN_ALIASES.value:
            return Objective._MIN_STR.value
        if self.value in Objective._MAX_ALIASES.value:
            return Objective._MAX_STR.value


# =============================================================================
# DATA CLASS
# =============================================================================

# converter
def _as_df(df):
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def _as_objective_array(arr):
    return np.array([Objective.construct_from_alias(a) for a in arr])


def _as_float_array(arr):
    return np.array(arr, dtype=float)


@attr.s(frozen=True, repr=False, cmp=False)
class DecisionMatrix:

    _data_df: pd.DataFrame = attr.ib(converter=_as_df)
    _objectives: np.ndarray = attr.ib(converter=_as_objective_array)
    _weights: np.ndarray = attr.ib(converter=_as_float_array)

    def __attrs_post_init__(self):
        _, c_number = np.shape(self._data_df)
        lens = {
            "c_number": c_number,
            "objectives": len(self._objectives),
            "weights": len(self._weights),
        }
        if len(set(lens.values())) > 1:
            del lens["c_number"]
            raise ValueError(
                "'objectives' and 'weights' must have the same number of "
                f"columns in 'data_df {c_number}. Found {lens}."
            )

    # CUSTOM CONSTRUCTORS =====================================================

    @classmethod
    def from_mcda_data(
        cls, mtx, objectives, weights=None, anames=None, cnames=None
    ):
        # first we need the number of alternatives and criteria
        try:
            a_number, c_number = np.shape(mtx)
        except ValueError:
            mtx_ndim = np.ndim(mtx)
            raise ValueError(
                f"'mtx' must have 2 dimensions, found {mtx_ndim} instead"
            )

        anames = np.asarray(
            [f"A{idx}" for idx in range(a_number)]
            if anames is None
            else anames
        )
        if len(anames) != a_number:
            raise ValueError(f"'anames' must have {a_number} elements")

        cnames = np.asarray(
            [f"C{idx}" for idx in range(c_number)]
            if cnames is None
            else cnames
        )

        if len(cnames) != c_number:
            raise ValueError(f"'cnames' must have {c_number} elements")

        weights = np.asarray(np.ones(c_number) if weights is None else weights)
        data_df = pd.DataFrame(mtx, index=anames, columns=cnames)

        return cls(data_df=data_df, objectives=objectives, weights=weights)

    @property
    def anames(self):
        return self._data_df.index.to_numpy()

    @property
    def cnames(self):
        return self._data_df.columns.to_numpy()

    @property
    def mtx(self):
        return self._data_df.to_numpy()

    @property
    def weights(self):
        return np.copy(self._weights)

    @property
    def objectives_values(self):
        return np.array([o.value for o in self._objectives], dtype=int)

    @property
    def objectives(self):
        return np.copy(self._objectives)

    @property
    def dtypes(self):
        return self.data_df.dtypes

    # utilities and redefinitions

    def copy(self, deep=True):
        return DecisionMatrix(
            data_df=self._data_df,
            objectives=self._objectives,
            weights=self._weights,
        )

    def __eq__(self, other):
        return (
            isinstance(other, DecisionMatrix)
            and self._data_df.equals(other._data_df)
            and np.array_equal(self._objectives, other._objectives)
            and np.array_equal(self._weights, other._weights)
        )

    def __ne__(self, other):
        return not self == other


# =============================================================================
# factory
# =============================================================================

mkdm = DecisionMatrix.from_mcda_data
