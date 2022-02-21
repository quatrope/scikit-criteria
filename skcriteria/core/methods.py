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


class SKCMethodABC(metaclass=abc.ABCMeta):
    """Base class for all class in scikit-criteria.

    Notes
    -----
    All estimators should specify:

    - ``_skcriteria_dm_type``: The type of the decision maker.
    - ``_skcriteria_parameters``: Availebe parameters.
    - ``_skcriteria_abstract_class``: If the class is abstract.

    If the class is *abstract* the user can ignore the other two attributes.

    """

    _skcriteria_abstract_class = True

    def __init_subclass__(cls):
        """Validate if the subclass are well formed."""
        is_abstract = vars(cls).get("_skcriteria_abstract_class", False)
        if is_abstract:
            return

        decisor_type = getattr(cls, "_skcriteria_dm_type", None)
        if decisor_type is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_dm_type'")
        cls._skcriteria_dm_type = str(decisor_type)

        params = getattr(cls, "_skcriteria_parameters", None)
        if params is None:
            raise TypeError(f"{cls} must redefine '_skcriteria_parameters'")

        params = frozenset(params)

        signature = inspect.signature(cls.__init__)
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )

        params_not_in_signature = params.difference(signature.parameters)
        if params_not_in_signature and not has_kwargs:

            raise TypeError(
                f"{cls} defines the parameters {params_not_in_signature} "
                "which is not found as a parameter in the __init__ method."
            )

        cls._skcriteria_parameters = params

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
# SKCTransformer ABC
# =============================================================================


class SKCTransformerABC(SKCMethodABC):
    """Abstract class for all transformer in scikit-criteria."""

    _skcriteria_dm_type = "transformer"
    _skcriteria_abstract_class = True

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

    The Transformer that implements this abstract class can be configured to
    transform
    `weights`, `matrix` or `both` so only that part of the DecisionMatrix
    is altered.

    This abstract class require to redefine ``_transform_weights`` and
    ``_transform_matrix``, instead of ``_transform_data``.

    """

    _skcriteria_abstract_class = True
    _skcriteria_parameters = ["target"]

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
