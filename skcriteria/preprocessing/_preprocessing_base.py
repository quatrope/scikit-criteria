#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities to create transformers."""

# =============================================================================
# IMPORTS
# =============================================================================

import abc

from ..core import DecisionMatrix, SKCMethodABC
from ..utils import doc_inherit


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


# =============================================================================
# MATRIX & WEIGHTS TRANSFORMER
# =============================================================================


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
