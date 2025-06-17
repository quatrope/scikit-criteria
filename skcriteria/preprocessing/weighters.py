#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for weight the criteria.

In addition to the main functionality, an MCDA agnostic function is offered
to calculate weights to a matrix along an arbitrary axis.


"""

# =============================================================================
# IMPORTS
# =============================================================================

from ..utils import hidden

with hidden():
    import abc
    import warnings

    import numpy as np

    import pandas as pd

    import scipy.stats

    from ._preprocessing_base import SKCTransformerABC
    from .scalers import matrix_scale_by_cenit_distance
    from ..core import Objective
    from ..utils import deprecated, doc_inherit

# =============================================================================
# BASE CLASS
# =============================================================================


class SKCWeighterABC(SKCTransformerABC):
    """Abstract class capable of determine the weights of the matrix.

    This abstract class require to redefine ``_weight_matrix``, instead of
    ``_transform_data``.

    """

    _skcriteria_abstract_class = True

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
# SAME WEIGHT
# =============================================================================


def equal_weights(matrix, base_value=1):
    r"""Use the same weights for all criteria.

    The result values are normalized by the number of columns.

    .. math::

        w_j = \frac{base\_value}{m}

    Where $m$ is the number os columns/criteria in matrix.


    Parameters
    ----------
    matrix: :py:class:`numpy.ndarray` like.
        The matrix of alternatives on which to calculate weights.
    base_value: int or float.
        Value to be normalized by the number of criteria to create the weights.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of weights

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import equal_weights
        >>> mtx = [[1, 2], [3, 4]]

        >>> equal_weights(mtx)
        array([0.5, 0.5])

    """
    ncriteria = np.shape(matrix)[1]
    weights = base_value / ncriteria
    return np.full(ncriteria, weights, dtype=float)


class EqualWeighter(SKCWeighterABC):
    """Assigns the same weights to all criteria.

    The algorithm calculates the weights as the ratio of ``base_value`` by the
    total criteria.

    """

    _skcriteria_parameters = ["base_value"]

    def __init__(self, base_value=1.0):
        self._base_value = float(base_value)

    @property
    def base_value(self):
        """Value to be normalized by the number of criteria."""
        return self._base_value

    @doc_inherit(SKCWeighterABC._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return equal_weights(matrix, self.base_value)


# =============================================================================
#
# =============================================================================


def std_weights(matrix):
    r"""Calculate weights as the standard deviation of each criterion.

    The result is normalized by the number of columns.

    .. math::

        w_j = \frac{s_j}{m}

    Where $m$ is the number os columns/criteria in matrix.

    Parameters
    ----------
    matrix: :py:class:`numpy.ndarray` like.
        The matrix of alternatives on which to calculate weights.

    Returns
    -------
    :py:class:`numpy.ndarray`
        array of weights

    Examples
    --------
    .. code-block:: pycon

        >>> from skcriteria.preprocess import std_weights
        >>> mtx = [[1, 2], [3, 4]]

        >>> std_weights(mtx)
         array([0.5, 0.5])

    """
    std = np.std(matrix, axis=0, ddof=1)
    return std / np.sum(std)


class StdWeighter(SKCWeighterABC):
    """Set as weight the normalized standard deviation of each criterion."""

    _skcriteria_parameters = []

    @doc_inherit(SKCWeighterABC._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return std_weights(matrix)


# =============================================================================
#
# =============================================================================


def entropy_weights(matrix):
    """Calculate the weights as the complement of the entropy of each \
    criterion.

    It uses the underlying ``scipy.stats.entropy`` function which assumes that
    the values of the criteria are probabilities of a distribution.

    The logarithmic base to use is the number of rows/alternatives in the
    matrix.

    This routine will normalize the sum of the weights to 1.

    See Also
    --------
    scipy.stats.entropy :
        Calculate the entropy of a distribution for given probability values.

    """
    base = len(matrix)
    entropy = scipy.stats.entropy(matrix, base=base, axis=0)
    entropy_divergence = 1 - entropy
    return entropy_divergence / np.sum(entropy_divergence)


class EntropyWeighter(SKCWeighterABC):
    """Assigns the complement of the entropy of the criteria as weights.

    It uses the underlying ``scipy.stats.entropy`` function which assumes that
    the values of the criteria are probabilities of a distribution.

    The logarithmic base to use is the number of rows/alternatives in the
    matrix.

    This transformer will normalize the sum of the weights to 1.

    See Also
    --------
    scipy.stats.entropy :
        Calculate the entropy of a distribution for given probability values.

    """

    _skcriteria_parameters = []

    @doc_inherit(SKCWeighterABC._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return entropy_weights(matrix)


# =============================================================================
#
# =============================================================================


@deprecated(
    reason="Please use ``pd.DataFrame(arr.T).correlation('pearson')``",
    version="0.8",
)
def pearson_correlation(arr):
    """Return Pearson product-moment correlation coefficients.

    This function is a thin wrapper of ``numpy.corrcoef``.

    Parameters
    ----------
    arr: array like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of arr represents a variable, and each column a single
        observation of all those variables.

    Returns
    -------
    R: numpy.ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    numpy.corrcoef :
        Return Pearson product-moment correlation coefficients.

    """
    return np.corrcoef(arr)


@deprecated(
    reason="Please use ``pd.DataFrame(arr.T).correlation('spearman')``",
    version="0.8",
)
def spearman_correlation(arr):
    """Calculate a Spearman correlation coefficient.

    This function is a thin wrapper of ``scipy.stats.spearmanr``.

    Parameters
    ----------
    arr: array like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of arr represents a variable, and each column a single
        observation of all those variables.

    Returns
    -------
    R: numpy.ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    scipy.stats.spearmanr :
        Calculate a Spearman correlation coefficient with associated p-value.

    """
    return scipy.stats.spearmanr(arr.T, axis=0).correlation


def critic_weights(matrix, objectives, correlation="pearson", scale=True):
    """Execute the CRITIC method without any validation."""
    # The paper:
    #   Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
    #   Determining objective weights in multiple criteria problems:
    #   The critic method. Computers & Operations Research, 22(7), 763-770.

    # and equation 1 of the paper
    matrix = np.asarray(matrix, dtype=float)

    # equation 2 an 3 of the paper
    matrix = (
        matrix_scale_by_cenit_distance(matrix, objectives=objectives)
        if scale
        else matrix
    )

    # equation 4
    corr = pd.DataFrame(matrix).corr(method=correlation).to_numpy(copy=True)
    one_minus_corr = 1 - corr

    # equation 5
    dindex = np.std(matrix, axis=0)
    uweights = dindex * np.sum(one_minus_corr, axis=0)

    # equation 6
    weights = uweights / np.sum(uweights)
    return weights


class CRITIC(SKCWeighterABC):
    """CRITIC (CRiteria Importance Through Intercriteria Correlation).

    The method aims at the determination of objective weights of relative
    importance in MCDM problems. The weights derived incorporate both contrast
    intensity and conflict which are contained in the structure of the decision
    problem.

    Parameters
    ----------
    correlation: str ["pearson", "spearman", "kendall"] or callable.
        This is the correlation function used to evaluate the discordance
        between two criteria. In other words, what conflict does one criterion
        a criterion with  respect to the decision made by the other criteria.
        By default the ``pearson`` correlation is used, and the ``spearman``
        and ``kendall`` correlation is also available implemented.
        It is also possible to provide a callable with input two 1d arrays
        and returning a float. Note that the returned matrix from corr will
        have 1 along the diagonals and will be symmetric regardless of the
        callable's behavior

    scale: bool (default ``True``)
        True if it is necessary to scale the data with
        ``skcriteria.preprocessing.matrix_scale_by_cenit_distance`` prior
        to calculating the correlation

    Warnings
    --------
    UserWarning:
        If some objective is to minimize. The original paper only suggests
        using it against maximization criteria, but there is no real
        mathematical constraint to use it for minimization.

    References
    ----------
    :cite:p:`diakoulaki1995determining`

    """

    CORRELATION = ("pearson", "spearman", "kendall")

    _skcriteria_parameters = ["correlation", "scale"]

    def __init__(self, correlation="pearson", scale=True):
        if not (correlation in self.CORRELATION or callable(correlation)):
            corr_keys = ", ".join(f"'{c}'" for c in self.CORRELATION)
            raise ValueError(f"Correlation must be {corr_keys} or a callable")
        self._correlation = correlation

        self._scale = bool(scale)

    @property
    def scale(self):
        """Return if it is necessary to scale the data."""
        return self._scale

    @property
    def correlation(self):
        """Correlation function."""
        return self._correlation

    @doc_inherit(SKCWeighterABC._weight_matrix)
    def _weight_matrix(self, matrix, objectives, **kwargs):
        if Objective.MIN.value in objectives:
            warnings.warn(
                "Although CRITIC can operate with minimization objectives, "
                "this is not recommended. Consider reversing the weights "
                "for these cases."
            )

        return critic_weights(
            matrix, objectives, correlation=self.correlation, scale=self.scale
        )


@deprecated(
    reason="Use ``skcriteria.preprocessing.weighters.CRITIC`` instead",
    version="0.8",
)
@doc_inherit(CRITIC, warn_class=False)
class Critic(CRITIC):
    pass


def gini_weights(matrix):
    r"""
    Calculates weights using the Gini coefficient.

    Computes the weights for each criterion (column) of the input matrix by
    calculating the Gini coefficient of each column, then normalizing those
    values to sum to 1.

    The columns are sorted to use the more efficient formula for the
    Gini coefficient:

    .. math::

        G = \frac{1}{n} \left( n + 1 - 2 \cdot \frac{
        \sum_{i=1}^n \left( \sum_{j=1}^i x_j \right)
        }{
        \sum_{i=1}^n x_i
        } \right)
    """
    n = matrix.shape[0]

    sorted_columns = np.sort(matrix, axis=0)
    cumulative_sums = np.cumsum(sorted_columns, axis=0)
    column_totals = cumulative_sums[-1]
    cumulative_totals = np.sum(cumulative_sums, axis=0)

    gini = (n + 1 - 2 * cumulative_totals / column_totals) / n

    return gini / np.sum(gini)


class GiniWeighter(SKCWeighterABC):
    """
    Calculates the weights with the Gini coefficient.

    The method aims at the determination of objective weights of relative
    importance in MCDM problems. It uses the Gini coefficient of the data of
    each criterion to assign the weights, giving a higher weight to a more
    unequal distribution. It takes the decision matrix as a parameter.
    See:
        G. Li and G. Chi, "A New Determining Objective Weights Method-Gini
        Coefficient Weight," 2009 First International Conference on
        Information Science and Engineering, Nanjing, China, 2009,
        pp. 3726-3729, doi: 10.1109/ICISE.2009.84.
    """

    _skcriteria_parameters = []

    @doc_inherit(SKCWeighterABC._weight_matrix)
    def _weight_matrix(self, matrix, **kwargs):
        return gini_weights(matrix)
