#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for scale values based on different strategies.

In addition to the Transformers, a collection of an MCDA agnostic functions
are offered to scale an array along an arbitrary axis.

"""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
from numpy import linalg

from sklearn import preprocessing as _sklpreproc

from ._preprocessing_base import (
    SKCMatrixAndWeightTransformerABC,
    SKCTransformerABC,
)
from ..core import Objective
from ..utils import deprecated, doc_inherit


# =============================================================================
# HELPER FUNCTION
# =============================================================================


def _run_sklearn_scaler(mtx_or_weights, scaler):
    """Runs sklearn scalers against 1D (weights) or 2D (alternatives) \
    arrays.

    This function is in charge of verifying if the array provided has adequate
    dimensions to work with the scikit-learn scalers.

    It also ensures that the output has the same input dimensions.

    """
    ndims = np.ndim(mtx_or_weights)
    if ndims == 1:  # is a weights
        mtx_or_weights = mtx_or_weights.reshape(len(mtx_or_weights), 1)
    result = scaler.fit_transform(mtx_or_weights)
    if ndims == 1:
        result = result.flatten()
    return result


# =============================================================================
# STANDAR SCALER
# =============================================================================


class StandarScaler(SKCMatrixAndWeightTransformerABC):
    """Standardize the dm by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the values, and `s` is the standard deviation
    of the training samples or one if `with_std=False`.

    This is a thin wrapper around ``sklearn.preprocessing.StandarScaler``.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently, unit
        standard deviation).

    """

    _skcriteria_parameters = ["target", "with_mean", "with_std"]

    def __init__(self, target, *, with_mean=True, with_std=True):
        super().__init__(target)
        self._with_mean = bool(with_mean)
        self._with_std = bool(with_std)

    @property
    def with_mean(self):
        """True if the features will be center before scaling."""
        return self._with_mean

    @property
    def with_std(self):
        """True if the features will be scaled to the unit variance."""
        return self._with_std

    def _get_scaler(self):
        return _sklpreproc.StandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std,
        )

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        scaler = self._get_scaler()
        return _run_sklearn_scaler(weights, scaler)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        scaler = self._get_scaler()
        return _run_sklearn_scaler(matrix, scaler)


# =============================================================================
# MINMAX
# =============================================================================


class MinMaxScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scaler based on the range.

    The matrix transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    And the weight transformation::

        X_std = (X - X.min(axis=None)) / (X.max(axis=None) - X.min(axis=None))
        X_scaled = X_std * (max - min) + min

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the range of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the range the weights.

    This is a thin wrapper around ``sklearn.preprocessing.MinMaxScaler``.

    Parameters
    ----------
    criteria_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `criteria_range`.

    """

    _skcriteria_parameters = ["target", "clip", "criteria_range"]

    def __init__(self, target, *, clip=False, criteria_range=(0, 1)):
        super().__init__(target)
        self._clip = bool(clip)
        self._cr_min, self._cr_max = map(float, criteria_range)

    @property
    def clip(self):
        """True if the transformed values will be clipped to held-out the \
        value provided `criteria_range`."""
        return self._clip

    @property
    def criteria_range(self):
        """Range of transformed data."""
        return (self._cr_min, self._cr_max)

    def _get_scaler(self):
        return _sklpreproc.MinMaxScaler(
            clip=self.clip,
            feature_range=self.criteria_range,
        )

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        scaler = self._get_scaler()
        return _run_sklearn_scaler(weights, scaler)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        scaler = self._get_scaler()
        return _run_sklearn_scaler(matrix, scaler)


# =============================================================================
# MAX
# =============================================================================


class MaxAbsScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scaler based on the maximum values.

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the maximum value of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the maximum value the weights.

    This estimator scales and translates each criteria individually such that
    the maximal absolute value of each criteria in the training set will be
    1.0. It does not shift/center the data, and thus does not destroy any
    sparsity.

    This is a thin wrapper around ``sklearn.preprocessing.MaxAbsScaler``.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        scaler = _sklpreproc.MaxAbsScaler()
        return _run_sklearn_scaler(weights, scaler)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        scaler = _sklpreproc.MaxAbsScaler()
        return _run_sklearn_scaler(matrix, scaler)


@deprecated(
    reason="Use ``skcriteria.preprocessing.scalers.MaxAbsScaler`` instead",
    version=0.8,
)
class MaxScaler(MaxAbsScaler):
    r"""Scaler based on the maximum values.

    From skcriteria >= 0.8 this is a thin wrapper around
    ``sklearn.preprocessing.MaxAbsScaler``.

    """


# =============================================================================
# VECTOR SCALER
# =============================================================================


def scale_by_vector(arr, axis=None):
    r"""Divide the array by norm of values defined vector along an axis.

    Calculates the set of ratios as the square roots of the sum of squared
    responses of a given axis as denominators.  If *axis* is *None* sum all
    the array.

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_vector
        >>> mtx = [[1, 2], [3, 4]]

        # ratios with the vector value of the array
        >>> scale_by_vector(mtx)
        array([[ 0.18257418,  0.36514837],
               [ 0.54772252,  0.73029673]])

        # ratios by column
        >>> scale_by_vector(mtx, axis=0)
        array([[ 0.31622776,  0.44721359],
               [ 0.94868326,  0.89442718]])

        # ratios by row
        >>> scale_by_vector(mtx, axis=1)
        array([[ 0.44721359,  0.89442718],
               [ 0.60000002,  0.80000001]])

    """
    arr = np.asarray(arr, dtype=float)
    frob = linalg.norm(arr, None, axis=axis)
    return arr / frob


class VectorScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scaler based on the norm of the vector..

    .. math::

        \overline{X}_{ij} =
        \frac{X_{ij}}{\sqrt{\sum\limits_{j=1}^m X_{ij}^{2}}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the norm of the vector defined by the values
    of that criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the vector defined by the values
    of the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_vector(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_vector(matrix, axis=0)


# =============================================================================
# SUM
# =============================================================================


def scale_by_sum(arr, axis=None):
    r"""Divide of every value on the array by sum of values along an axis.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    Parameters
    ----------
    arr: :py:class:`numpy.ndarray` like.
        A array with values
    axis : :py:class:`int` optional
        Axis along which to operate.  By default, flattened input is used.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of ratios

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import scale_by_sum
        >>> mtx = [[1, 2], [3, 4]]

        >>> scale_by_sum(mtx) # ratios with the sum of the array
        array([[ 0.1       ,  0.2       ],
               [ 0.30000001,  0.40000001]])

        # ratios with the sum of the array by column
        >>> scale_by_sum(mtx, axis=0)
        array([[ 0.25      ,  0.33333334],
               [ 0.75      ,  0.66666669]])

        # ratios with the sum of the array by row
        >>> scale_by_sum(mtx, axis=1)
        array([[ 0.33333334,  0.66666669],
               [ 0.42857143,  0.5714286 ]])

    """
    arr = np.asarray(arr, dtype=float)
    sumval = np.sum(arr, axis=axis, keepdims=True)
    return arr / sumval


class SumScaler(SKCMatrixAndWeightTransformerABC):
    r"""Scalerbased on the total sum of values.

    .. math::

        \overline{X}_{ij} = \frac{X_{ij}}{\sum\limits_{j=1}^m X_{ij}}

    If the scaler is configured to work with 'matrix' each value
    of each criteria is divided by the total sum of all the values of that
    criteria.
    In other hand if is configure to work with 'weights',
    each value of weight is divided by the total sum of all the weights.

    """

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_weights)
    def _transform_weights(self, weights):
        return scale_by_sum(weights, axis=None)

    @doc_inherit(SKCMatrixAndWeightTransformerABC._transform_matrix)
    def _transform_matrix(self, matrix):
        return scale_by_sum(matrix, axis=0)


# =============================================================================
# CENIT DISTANCE
# =============================================================================


def matrix_scale_by_cenit_distance(matrix, objectives):
    r"""Calculate a scores with respect to an ideal and anti-ideal alternative.

    For every criterion :math:`f` of this multicriteria problem we define a
    membership function :math:`x_j` mapping the values of :math:`f_j` to the
    interval [0, 1].

    The result score :math:`x_{aj}`expresses the degree to which the
    alternative  :math:`a` is close to the ideal value :math:`f_{j}^*`, which
    is the best performance in criterion , and  far from the anti-ideal value
    :math:`f_{j^*}`, which is the worst performance in  criterion :math:`j`.
    Both ideal and anti-ideal, are achieved by at least one of the alternatives
    under consideration.

    .. math::

        x_{aj} = \frac{f_j(a) - f_{j^*}}{f_{j}^* - f_{j^*}}

    """
    matrix = np.asarray(matrix, dtype=float)

    maxs = np.max(matrix, axis=0)
    mins = np.min(matrix, axis=0)

    where_max = np.equal(objectives, Objective.MAX.value)

    cenit = np.where(where_max, maxs, mins)
    nadir = np.where(where_max, mins, maxs)

    return (matrix - nadir) / (cenit - nadir)


class CenitDistanceMatrixScaler(SKCTransformerABC):
    r"""Relative scores with respect to an ideal and anti-ideal alternative.

    For every criterion :math:`f` of this multicriteria problem we define a
    membership function :math:`x_j` mapping the values of :math:`f_j` to the
    interval [0, 1].

    The result score :math:`x_{aj}`expresses the degree to which the
    alternative  :math:`a` is close to the ideal value :math:`f_{j}^*`, which
    is the best performance in criterion , and  far from the anti-ideal value
    :math:`f_{j^*}`, which is the worst performance in  criterion :math:`j`.
    Both ideal and anti-ideal, are achieved by at least one of the alternatives
    under consideration.

    .. math::

        x_{aj} = \frac{f_j(a) - f_{j^*}}{f_{j}^* - f_{j^*}}


    References
    ----------
    :cite:p:`diakoulaki1995determining`

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, objectives, **kwargs):

        distance_mtx = matrix_scale_by_cenit_distance(matrix, objectives)

        dtypes = np.full(np.shape(objectives), float)

        kwargs.update(
            matrix=distance_mtx, objectives=objectives, dtypes=dtypes
        )
        return kwargs
