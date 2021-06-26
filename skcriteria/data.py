# el tipo de datos Data de skcriteria, debe contener:
# - La matriz de alternativas (mtx).
# - El sentido de los criterios (sense)
# - Pesos (weights)
# - nombre de los atributos (anames)
# - nombre de los criterios (cnames)

# notes
# https://github.com/tomharvey/pandas-extension-dtype/blob/master/decimal_array.py
# https://github.com/0phoff/pygeos-pandas/blob/master/pgpd/_array.py
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extension-types

# =============================================================================
# IMPORTS
# =============================================================================

import enum
import numbers
from collections.abc import Iterable

import attr

import numpy as np

import pandas as pd
from pandas.api import extensions as pdext


# =============================================================================
# OBJECTIVE ENUM
# =============================================================================


class Objective(enum.Enum):
    MIN = -1
    MAX = 1

    _MIN_STR = "\u25bc"
    _MAX_STR = "\u25b2"

    #: Another way to name the maximization criteria.
    _MAX_ALIASES = [
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

    #: Another ways to name the minimization criteria.
    _MIN_ALIASES = [
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

    def __str__(self):
        return self.name

    def to_string(self):
        if self.value in Objective._MIN_ALIASES.value:
            return Objective._MIN_STR.value
        if self.value in Objective._MAX_ALIASES.value:
            return Objective._MAX_STR.value

    @classmethod
    def construct_from_alias(cls, alias):
        if isinstance(alias, str):
            alias = alias.lower()
        if alias in cls._MAX_ALIASES.value:
            return cls.MAX
        if alias in cls._MIN_ALIASES.value:
            return cls.MIN
        raise ValueError(f"Invalid criteria objective {alias}")


# =============================================================================
# CUSTOM DTYPE
# =============================================================================
@pdext.register_extension_dtype
@attr.s(repr=False, hash=True)
class CriteriaDtype(pdext.ExtensionDtype):
    """A custom data type, to be paired with an ExtensionArray."""

    # dtype related
    type = np.number
    name = "criteria"
    na_value = pd.NA

    _is_numeric = True
    _metadata = ("objective", "weight")

    # data itself
    objective = attr.ib(converter=Objective.construct_from_alias)
    weight = attr.ib(converter=float)

    def __attrs_post_init__(self):
        # we rewrite the name in the instance so the objective and weight
        # is shown when the array is put in a series
        self.name = f"{self.name}({self.objective}, {self.weight})"

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}({self.objective}, {self.weight})"

    @classmethod
    def construct_from_string(cls, string):
        """
        Construct this type from a string (ic. :attr:`~CriteriaDtype.name`).
        Args:
            string (str): The name of the type.
        Returns:
            CriteriaDtype: instance of the dtype.
        Raises:
            TypeError: string is not equal to "criteria".
        """
        if string == cls.name:
            return cls()
        else:
            raise TypeError(
                f'Cannot construct a "{cls.__name__}" from "{string}"'
            )

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return CriteriaArray


# =============================================================================
# CUSTOM ARRAY
# =============================================================================


@attr.s(repr=False)
class CriteriaArray(pdext.ExtensionArray):

    data = attr.ib(converter=np.asarray)
    _dtype = attr.ib(validator=attr.validators.instance_of(CriteriaDtype))

    # INTERNAL

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        cls_name = type(self).__name__
        return f"{cls_name}({self.data}, {self.dtype})"

    @property
    def dtype(self):
        return self._dtype

    # =========================================================================
    # ExtensionArray Specific
    # =========================================================================

    @classmethod
    def _from_sequence(cls, scalars, dtype, copy=False):
        values = np.asarray(scalars)
        if copy:
            values = values.copy()
        return cls(data=values, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization."""
        return cls(values)

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self.data[key]

        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(key, (Iterable, slice)):
            return CriteriaArray(self.data[key], dtype=self.dtype)
        else:
            raise TypeError("Index type not supported", key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.data == other.data
            and self.dtype == other.dtype
        )

    @property
    def nbytes(self):
        return self.data.nbytes

    def isna(self):
        return pd.isna(self.data)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.core.algorithms import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif not isinstance(fill_value, self.dtype.type):
                raise TypeError("Provide float or None as fill value")

        result = take(
            self.data, indices, allow_fill=allow_fill, fill_value=fill_value
        )

        if allow_fill and fill_value is None:
            result[pd.isna(result)] = None

        return self.__class__(result)

    def copy(self, order="C"):
        cdata = self.data.copy(order)
        return CriteriaArray(cdata, dtype=self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([c.data for c in to_concat])
        dtype = {c.dtype for c in to_concat}
        if len(dtype) > 1:
            raise ValueError("Multiple criteria dtype detected")
        return cls(data, dtype=dtype.pop())


# =============================================================================
# SERIES AND DATA FRAME
# =============================================================================


class CriteriaSeries(pd.Series):
    @property
    def _constructor(self):
        return CriteriaSeries

    @property
    def _constructor_expanddim(self):
        return DMFrame


# =============================================================================
# SERIES DATA FRAME
# =============================================================================
class DMFrame(pd.DataFrame):

    # Pandas extension API ====================================================
    @property
    def _constructor(self):
        return DMFrame

    @property
    def _constructor_sliced(self):
        return CriteriaSeries

    # Custom constructors =====================================================

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

        cnames = np.asarray(
            [f"C{idx}" for idx in range(c_number)]
            if cnames is None
            else cnames
        )

        weights = np.asarray(np.ones(c_number) if weights is None else weights)

        # validations of longitudes of criteria
        # in python >= 3.1 we use zip(..., strict=True)
        lens = {
            "c_number": c_number,
            "cnames": len(cnames),
            "objectives": len(objectives),
            "weights": len(weights),
        }
        if len(set(lens.values())) > 1:
            del lens["c_number"]
            raise ValueError(
                "'cnames', 'objectives' and 'weights' must have "
                f"{c_number} values. Found {lens}"
            )

        dtypes = {}
        for cname, objective, weight in zip(cnames, objectives, weights):
            dtype = CriteriaDtype(objective=objective, weight=weight)
            dtypes[cname] = dtype

        df = cls(data=mtx, index=anames, columns=cnames)
        mcdf = df.astype(dtypes, copy=False)

        return mcdf

    # CUSTOM reimplmentations =================================================

    def to_string(self, **kwargs):
        cow = zip(self.cnames, self.objectives_dtype, self.weights)
        kwargs["header"] = [
            f"{c}[{obj.to_string()} {weight}]" for c, obj, weight in cow
        ]
        return super().to_string(**kwargs)

    # MCDA ====================================================================

    @property
    def ctypes(self):
        def extract_ctypes(v):
            return v.values.data.dtype

        return self.apply(extract_ctypes, axis=0).to_numpy()

    @property
    def anames(self):
        return self.index.to_numpy()

    @property
    def cnames(self):
        return self.columns.to_numpy()

    @property
    def mtx(self):
        return self.to_numpy()

    @property
    def weights(self):
        def extract_weights(v):
            return v.weight

        return self.dtypes.apply(extract_weights).to_numpy()

    @property
    def objectives_dtype(self):
        def extract_objectives_dtype(v):
            return v.objective

        return self.dtypes.apply(extract_objectives_dtype).to_numpy()

    @property
    def objectives(self):
        def extract_objectives(v):
            return v.objective.value

        return self.dtypes.apply(extract_objectives).to_numpy()


# =============================================================================
# factory
# =============================================================================

mkdm = DMFrame.from_mcda_data
