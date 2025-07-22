#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Implementation of functionalities for convert minimization criteria into \
maximization ones."""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
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
        # only the changed columns are changed
        columns_changes = np.any(inv_mtx != matrix, axis=0)
        inv_dtypes = np.where(columns_changes, inv_mtx.dtype, dtypes)

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
    version="0.7",
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


# =============================================================================
# MIN-MAX INVERSION
# =============================================================================


class MinMaxInverter(SKCObjectivesInverterABC):
    r"""Normalize and invert minimization criteria using min-max scaling.

    For minimization criteria, values are inverted by normalizing with:

        (x - max) / (min - max)

    which converts the minimization problem into a maximization one.

    For maximization criteria, values are normalized with:

        (x - min) / (max - min)

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCObjectivesInverterABC._invert)
    def _invert(self, matrix, minimize_mask):
        """Apply min-max normalization that inverts minimization criteria."""

        cost = minimize_mask
        benefit = ~cost

        inverted_matrix = np.empty_like(matrix)

        maxs = np.max(matrix, axis=0)
        mins = np.min(matrix, axis=0)

        inverted_matrix[:, cost] = (matrix[:, cost] - maxs[cost]) / (
            mins[cost] - maxs[cost]
        )
        inverted_matrix[:, benefit] = (matrix[:, benefit] - mins[benefit]) / (
            maxs[benefit] - mins[benefit]
        )

        return inverted_matrix


# =============================================================================
# BENEFIT-COST INVERSION
# =============================================================================


class BenefitCostInverter(SKCObjectivesInverterABC):
    r"""Inverts using ratios based on criterion type.

    The matrix transformation is given by:

    .. math::

    For each criterion j, the normalized value is calculated as:
        if j is a benefit criteria:
            n_{ij} = \frac{x_{ij}}{\max_i x_{ij}}

        if j is a cost criteria:
            n_{ij} = \frac{\min_i x_{ij}}{x_{ij}}

    Raises
    ------
    ValueError:
        If the decision matrix contains negative values.

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCObjectivesInverterABC._invert)
    def _invert(self, matrix, minimize_mask):
        if np.any(matrix < 0):
            raise ValueError("Inverter can not operate with negative values")

        inv_mtx = np.zeros_like(matrix, dtype=float)

        # Benefit Criteria
        maximize_mask = ~minimize_mask
        max_columns = matrix[:, maximize_mask]
        max_values = np.max(max_columns, axis=0)
        # Avoid division by zero
        checked_max_values = np.where(max_values != 0, max_values, 1e-5)
        inv_mtx[:, maximize_mask] = max_columns / checked_max_values

        # Cost Criteria
        min_columns = matrix[:, minimize_mask]
        min_values = np.min(min_columns, axis=0)
        # Avoid division by zero
        checked_min_columns = np.where(min_columns != 0, min_columns, 1e-5)
        inv_mtx[:, minimize_mask] = min_values / checked_min_columns

        return inv_mtx
