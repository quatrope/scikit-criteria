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

import abc
import inspect

from .data import DecisionMatrix
from .utils import doc_inherit

# =============================================================================
# BASE DECISION MAKER CLASS
# =============================================================================


_IGNORE_PARAMS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


class SKCBaseDecisionMaker:
    """Base class for all class in scikit-criteria.

    Notes
    -----
    All estimators should specify:

    - ``_skcriteria_dm_type``: The type of the decision maker.


    """

    _skcriteria_dm_type = None
    _skcriteria_parameters = None

    def __init_subclass__(cls):
        """Validate if the subclass are well formed."""
        decisor_type = cls._skcriteria_dm_type

        if decisor_type is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_dm_type'")

        if (
            cls._skcriteria_parameters is None
            and cls.__init__ is not SKCBaseDecisionMaker.__init__
        ):
            signature = inspect.signature(cls.__init__)
            parameters = set()
            for idx, param_tuple in enumerate(signature.parameters.items()):
                if idx == 0:  # first arugment of a method is the instance
                    continue
                name, param = param_tuple
                if param.kind not in _IGNORE_PARAMS:
                    parameters.add(name)
            cls._skcriteria_parameters = frozenset(parameters)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        cls_name = type(self).__name__

        parameters = []
        if self._skcriteria_parameters:
            for pname in sorted(self._skcriteria_parameters):
                pvalue = getattr(self, pname)
                parameters.append(f"{pname}={repr(pvalue)}")

        str_parameters = ", ".join(parameters)
        return f"{cls_name}({str_parameters})"


# =============================================================================
# VALIDATOR MIXIN
# =============================================================================


class SKCDataValidatorMixin(metaclass=abc.ABCMeta):
    """Mixin class to add a _validate_data method."""

    @abc.abstractmethod
    def _validate_data(self, **kwargs):
        """Validate all the data previously to send to the real algorithm."""
        raise NotImplementedError()


# =============================================================================
# SKCTransformer MIXIN
# =============================================================================


class SKCTransformerMixin(SKCDataValidatorMixin, metaclass=abc.ABCMeta):
    """Mixin class for all transformer in scikit-criteria."""

    _skcriteria_dm_type = "transformer"

    @abc.abstractmethod
    def _transform_data(self, **kwargs):
        """Apply the transformation logic to the decision matrix parameters.

        Parameters
        ----------
        kwargs:
            The decision matrix as separated parameters.

        Returns
        -------
        :py:class:`dict`
            A dictionary with all the values of the decision matrix
            transformed.

        """
        raise NotImplementedError()

    def transform(self, dm):
        """Perform transformation on `dm`.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            The decision matrix to transform.

        Returns
        -------
        :py:class:`skcriteria.data.DecisionMatrix`
            Transformed decision matrix.

        """
        mtx = dm.matrix
        objectives = dm.objectives_values
        weights = dm.weights
        anames = dm.anames
        cnames = dm.cnames
        dtypes = dm.dtypes

        self._validate_data(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=dtypes,
        )

        transformed_kwargs = self._transform_data(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
            dtypes=dtypes,
        )

        transformed_dm = DecisionMatrix.from_mcda_data(**transformed_kwargs)

        return transformed_dm


class SKCMatrixAndWeightTransformerMixin(SKCTransformerMixin):
    """Transform weights and matrix together or independently.

    The Transformer that implements this mixin can be configured to transform
    `weights`, `matrix` or `both` so only that part of the DecisionMatrix
    is altered.

    This mixin require to redefine ``_transform_weights`` and
    ``_transform_matrix``, instead of ``_transform_data``.

    """

    _TARGET_WEIGHTS = "weights"
    _TARGET_MATRIX = "matrix"
    _TARGET_BOTH = "both"

    def __init__(self, target):
        self.target = target

    @property
    def target(self):
        """Determine which part of the DecisionMatrix will be transformed."""
        return self._target

    @target.setter
    def target(self, target):
        if target not in (
            self._TARGET_MATRIX,
            self._TARGET_WEIGHTS,
            self._TARGET_BOTH,
        ):
            raise ValueError(
                f"'target' can only be '{self._TARGET_WEIGHTS}' or "
                f"'{self._TARGET_MATRIX}'', found '{target}'"
            )
        self._target = target

    @abc.abstractmethod
    def _transform_weights(self, weights):
        """Execute the transform method over the weights.

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

    @abc.abstractmethod
    def _transform_matrix(self, matrix):
        """Execute the transform method over the matrix.

        Parameters
        ----------
        matrix: :py:class:`numpy.ndarray`
            The decision matrix to transform

        Returns
        -------
        :py:class:`numpy.ndarray`
            The transformed matrix.

        """
        raise NotImplementedError()

    @doc_inherit(SKCTransformerMixin._transform_data)
    def _transform_data(self, matrix, weights, **kwargs):
        norm_mtx = matrix
        norm_weights = weights

        if self._target in (self._TARGET_MATRIX, self._TARGET_BOTH):
            norm_mtx = self._transform_matrix(matrix)

        if self._target in (self._TARGET_WEIGHTS, self._TARGET_BOTH):
            norm_weights = self._transform_weights(weights)

        kwargs.update(matrix=norm_mtx, weights=norm_weights, dtypes=None)

        return kwargs


# =============================================================================
# SK WEIGHTER
# =============================================================================


class SKCWeighterMixin(SKCTransformerMixin):
    """Mixin capable of determine the weights of the matrix.

    This mixin require to redefine ``_weight_matrix``, instead of
    ``_transform_data``.

    """

    @abc.abstractmethod
    def _weight_matrix(self, matrix, objectives, weights):
        """Transform the matrix and return an array of weights.

        Parameters
        ----------
        matrix: :py:class:`numpy.ndarray`
            The decision matrix to weights.
        objectives: :py:class:`numpy.ndarray`
            The objectives in numeric format.
        weights: :py:class:`numpy.ndarray`
            The original weights

        Returns
        -------
        :py:class:`numpy.ndarray`
            An array of weights.

        """
        raise NotImplementedError()

    @doc_inherit(SKCTransformerMixin._transform_data)
    def _transform_data(self, matrix, objectives, weights, **kwargs):

        new_weights = self._weight_matrix(
            matrix=matrix, objectives=objectives, weights=weights
        )

        kwargs.update(
            matrix=matrix, objectives=objectives, weights=new_weights
        )

        return kwargs
