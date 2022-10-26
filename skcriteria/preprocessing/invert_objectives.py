#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for convert minimization criteria into \
maximization ones."""

# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np

from ._preprocessing_base import SKCTransformerABC
from ..core import Objective
from ..utils import deprecated, doc_inherit


# =============================================================================
# Base Class
# =============================================================================
class SKCObjectivesInverterABC(SKCTransformerABC):
    """Abstract class capable of invert objectives.

    This abstract class require to redefine ``_invert``, instead of
    ``_transform_data``.

    """

    _skcriteria_abstract_class = True

    def _invert(self, matrix, minimize_mask):
        """Invert the minimization objectives.

        Parameters
        ----------
        matrix: :py:class:`numpy.ndarray`
            The decision matrix to weights.
        minimize_mask: :py:class:`numpy.ndarray`
            Mask with the same size as the columns in the matrix. True values
            indicate that this column is a criterion to be minimized.

        Returns
        -------
        :py:class:`numpy.ndarray`
            A new matrix with the minimization objectives inverted.

        """
        raise NotImplementedError()

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, objectives, dtypes, **kwargs):
        # check where we need to transform
        minimize_mask = np.equal(objectives, Objective.MIN.value)

        # execute the transformation
        inv_mtx = self._invert(matrix, minimize_mask)

        # new objective array
        inv_objectives = np.full(
            len(objectives), Objective.MAX.value, dtype=int
        )

        # we are trying to preserve the original dtype as much as possible
        # only the minimize criteria are changed.
        inv_dtypes = np.where(minimize_mask, inv_mtx.dtype, dtypes)

        kwargs.update(
            matrix=inv_mtx, objectives=inv_objectives, dtypes=inv_dtypes
        )
        return kwargs


# =============================================================================
# -x
# =============================================================================
class NegateMinimize(SKCObjectivesInverterABC):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{-{C}}`.

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCObjectivesInverterABC._invert)
    def _invert(self, matrix, minimize_mask):
        inv_mtx = np.array(matrix, dtype=float)

        inverted_values = -inv_mtx[:, minimize_mask]
        inv_mtx[:, minimize_mask] = inverted_values

        return inv_mtx


# =============================================================================
# 1/x
# =============================================================================
class InvertMinimize(SKCObjectivesInverterABC):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    Notes
    -----
    All the dtypes of the decision matrix are preserved except the inverted
    ones thar are converted to ``numpy.float64``.

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCObjectivesInverterABC._invert)
    def _invert(self, matrix, minimize_mask):
        inv_mtx = np.array(matrix, dtype=float)

        inverted_values = 1.0 / inv_mtx[:, minimize_mask]
        inv_mtx[:, minimize_mask] = inverted_values

        return inv_mtx


# =============================================================================
# DEPRECATED
# =============================================================================
@deprecated(
    reason=(
        "Use ``skcriteria.preprocessing.invert_objectives.InvertMinimize`` "
        "instead"
    ),
    version=0.7,
)
class MinimizeToMaximize(InvertMinimize):
    r"""Transform all minimization criteria  into maximization ones.

    The transformations are made by calculating the inverse value of
    the minimization criteria. :math:`\min{C} \equiv \max{\frac{1}{C}}`

    Notes
    -----
    All the dtypes of the decision matrix are preserved except the inverted
    ones thar are converted to ``numpy.float64``.

    """
