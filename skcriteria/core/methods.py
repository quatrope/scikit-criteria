#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities of scikit-criteria."""

# =============================================================================
# IMPORTS
# =============================================================================ç

import abc
import copy
import inspect

from .data import DecisionMatrix
from ..utils import doc_inherit

# =============================================================================
# BASE DECISION MAKER CLASS
# =============================================================================


_INVALID_FOR_AUTO_PARAMS_INSPECTION = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


class SKCMethodABC(metaclass=abc.ABCMeta):
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

        if "_skcriteria_parameters" not in vars(cls):
            parameters = set()

            if cls.__init__ is not SKCMethodABC.__init__:
                signature = inspect.signature(cls.__init__)

                for idx, param_tuple in enumerate(
                    signature.parameters.items()
                ):
                    if idx == 0:  # first arugment of a method is the instance
                        continue
                    name, param = param_tuple
                    if param.kind in _INVALID_FOR_AUTO_PARAMS_INSPECTION:
                        raise TypeError(
                            "The automatic parameter inspection "
                            "does not work with variable parameters."
                        )

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

    def get_parameters(self):
        """Return the parameters of the method as dictionary."""
        the_parameters = {}
        for parameter_name in self._skcriteria_parameters:
            parameter_value = getattr(self, parameter_name)
            the_parameters[parameter_name] = copy.deepcopy(parameter_value)
        return the_parameters

    def copy(self, **kwargs):
        """Return a deep copy of the current Object..

        This method is also useful for manually modifying the values of the
        object.

        Parameters
        ----------
        kwargs :
            The same parameters supported by object constructor. The values
            provided replace the existing ones in the object to be copied.

        Returns
        -------
        A new object.

        """
        asdict = self.get_parameters()
        asdict.update(kwargs)

        cls = type(self)
        return cls(**asdict)


# =============================================================================
# SKCTransformer MIXIN
# =============================================================================


class SKCTransformerABC(SKCMethodABC):
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
        data = dm.to_dict()

        transformed_data = self._transform_data(**data)

        transformed_dm = DecisionMatrix.from_mcda_data(**transformed_data)

        return transformed_dm


class SKCMatrixAndWeightTransformerABC(SKCTransformerABC):
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
        if target not in (
            self._TARGET_MATRIX,
            self._TARGET_WEIGHTS,
            self._TARGET_BOTH,
        ):
            raise ValueError(
                f"'target' can only be '{self._TARGET_WEIGHTS}', "
                f"'{self._TARGET_MATRIX}' or '{self._TARGET_BOTH}', "
                f"found '{target}'"
            )
        self._target = target

    @property
    def target(self):
        """Determine which part of the DecisionMatrix will be transformed."""
        return self._target

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

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, weights, **kwargs):
        transformed_mtx = matrix
        transformed_weights = weights

        if self._target in (self._TARGET_MATRIX, self._TARGET_BOTH):
            transformed_mtx = self._transform_matrix(matrix)

        if self._target in (self._TARGET_WEIGHTS, self._TARGET_BOTH):
            transformed_weights = self._transform_weights(weights)

        kwargs.update(
            matrix=transformed_mtx, weights=transformed_weights, dtypes=None
        )

        return kwargs


# =============================================================================
# SK WEIGHTER
# =============================================================================


class SKCWeighterABC(SKCTransformerABC):
    """Mixin capable of determine the weights of the matrix.

    This mixin require to redefine ``_weight_matrix``, instead of
    ``_transform_data``.

    """

    @abc.abstractmethod
    def _weight_matrix(self, matrix, objectives, weights):
        """Calculate a new array of weights.

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

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, objectives, weights, **kwargs):

        new_weights = self._weight_matrix(
            matrix=matrix, objectives=objectives, weights=weights
        )

        kwargs.update(
            matrix=matrix, objectives=objectives, weights=new_weights
        )

        return kwargs


# =============================================================================
#
# =============================================================================


class SKCDecisionMakerABC(SKCMethodABC):
    """Mixin class for all decisor based methods in scikit-criteria."""

    _skcriteria_dm_type = "decision_maker"

    @abc.abstractmethod
    def _evaluate_data(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_result(self, alternatives, values, extra):
        raise NotImplementedError()

    def evaluate(self, dm):
        """Validate the dm and calculate and evaluate the alternatives.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the ranking will be calculated.

        Returns
        -------
        :py:class:`skcriteria.data.RankResult`
            Ranking.

        """
        data = dm.to_dict()

        result_data, extra = self._evaluate_data(**data)

        alternatives = data["alternatives"]
        result = self._make_result(
            alternatives=alternatives, values=result_data, extra=extra
        )

        return result
