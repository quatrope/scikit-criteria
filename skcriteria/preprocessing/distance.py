#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Normalization through the distance to distance function."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..core import Objective, SKCTransformerABC
from ..utils import doc_inherit

# =============================================================================
# FUNCTIONS
# =============================================================================


def cenit_distance(matrix, objectives):
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


class CenitDistance(SKCTransformerABC):
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

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, matrix, objectives, **kwargs):

        distance_mtx = cenit_distance(matrix, objectives)

        dtypes = np.full(np.shape(objectives), float)

        kwargs.update(
            matrix=distance_mtx, objectives=objectives, dtypes=dtypes
        )
        return kwargs
