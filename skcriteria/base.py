#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities of scikit-criteria."""

# =============================================================================
# IMPORTS
# =============================================================================ç

import inspect

import numpy as np

from .data import DecisionMatrix

# =============================================================================
# BASE DECISION MAKER CLASS
# =============================================================================


_IGNORE_PARAMS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


class BaseDecisionMaker:
    """Base class for all decision maker in scikit-criteria.

    Notes
    -----
    All estimators should specify:

    - ``_skcriteria_dm_type``: The type of the decision maker.


    """

    _skcriteria_dm_type = None
    _skcriteria_parameters = None

    def __init_subclass__(cls) -> None:
        """Validate if the subclass are well formed."""
        decisor_type = cls._skcriteria_dm_type

        if decisor_type is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_dm_type'")

        if (
            cls._skcriteria_parameters is None
            and cls.__init__ is not BaseDecisionMaker.__init__
        ):
            parameters = []

            signature = inspect.signature(cls.__init__)
            for idx, param_tuple in enumerate(signature.parameters.items()):
                if idx == 0:  # first arugment of a method is the instance
                    continue
                name, param = param_tuple
                if param.kind not in _IGNORE_PARAMS:
                    parameters.append(name)
            cls._skcriteria_parameters = frozenset(parameters)

    def __repr__(self) -> str:
        """x.__repr__() <==> repr(x)."""
        cls_name = type(self).__name__
        parameters = []
        for pname in self._skcriteria_parameters:
            pvalue = getattr(self, pname)
            parameters.append(f"{pname}={repr(pvalue)}")
        str_parameters = ", ".join(parameters)
        return f"{cls_name}({str_parameters})"

    def validate_data(self, **kwargs) -> None:
        """Validate all the data previously to send to the real algorithm."""
        pass


# =============================================================================
# NORMALIZER MIXIN
# =============================================================================


class NormalizerMixin:
    """Mixin class for all normalizers in scikit-criteria."""

    _skcriteria_dm_type = "normalizer"

    def normalize_data(self, **kwargs) -> dict:  # noqa: D401
        """Generic implementation of the normalizer logic.

        Parameters
        ----------
        kwargs:
            The decision matrix as separated parameters.

        Returns
        -------
        :py:class:`dict`
            A dictionary with all the values of the normalized decision matrix.
            This parameters will be provided into
            :py:method:`DecisionMatrix.from_mcda_data`.

        """
        raise NotImplementedError()

    def normalize(self, dm) -> DecisionMatrix:
        """Perform normalization on `dm` and returns normalized \
        version of it.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            The decision matrix to normalize.

        Returns
        -------
        :py:class:`skcriteria.data.DecisionMatrix`
            Normalized decision matrix.

        """
        mtx = dm.mtx
        objectives = dm.objectives_values
        weights = dm.weights
        anames = dm.anames
        cnames = dm.cnames
        dtypes = dm.dtypes

        self.validate_data(
            mtx=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=dtypes,
        )

        nkwargs = self.normalize_data(
            mtx=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=dtypes,
        )

        norm_dm = DecisionMatrix.from_mcda_data(**nkwargs)

        return norm_dm


class MatrixAndWeightNormalizerMixin(NormalizerMixin):

    FOR_WEIGHTS = "weights"
    FOR_MATRIX = "matrix"

    def __init__(self, normalize_for: str = "matrix") -> None:
        if normalize_for not in (self.FOR_MATRIX, self.FOR_WEIGHTS):
            raise ValueError(
                f"'normalize_for' can only be '{self.FOR_WEIGHTS}' or "
                f"'{self.FOR_MATRIX}'', found '{normalize_for}'"
            )
        self._normalize_for = normalize_for

    @property
    def normalize_for(self) -> str:
        return self._normalize_for

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Execute the normalize method over the matrix.

        Parameters
        ----------
        weights: :py:class:`numpy.ndarray`
            The weights to transform.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The transformed weights.

        """
        raise NotImplementedError()

    def normalize_matrix(self, mtx: np.ndarray) -> np.ndarray:
        """Execute the normalize method over the matrix.

        Parameters
        ----------
        mtx: :py:class:`numpy.ndarray`
            The decision matrix to transform

        Returns
        -------
        :py:class:`numpy.ndarray`
            The transformed matrix.

        """
        raise NotImplementedError()

    def normalize_data(
        self, mtx: np.ndarray, weights: np.ndarray, **kwargs
    ) -> dict:
        """Execute the transformation over the provided data.

        Returns
        -------
        :py:class:`dict`
            A dictionary with all the values of the normalized decision matrix.
            This parameters will be provided into
            :py:method:`DecisionMatrix.from_mcda_data`.

        """
        if self._normalize_for == self.FOR_MATRIX:
            norm_mtx = self.normalize_matrix(mtx)
            norm_weights = weights
        else:
            norm_mtx = mtx
            norm_weights = self.normalize_weights(weights)

        kwargs.update(
            mtx=norm_mtx, weights=norm_weights, dtypes=None
        )

        return kwargs
