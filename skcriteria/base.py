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
# =============================================================================


from .data import DecisionMatrix

# =============================================================================
# BASE DECISION MAKER CLASS
# =============================================================================


class BaseDecisionMaker:
    """Base class for all decision maker in scikit-criteria.

    Notes
    -----
    All estimators should specify:

    - ``_dm_type``: The type of the decision maker.


    """

    _dm_type = None
    _main_method_name = None

    def __init_subclass__(cls) -> None:
        """Validate if the subclass are well formed."""
        decisor_type = cls._dm_type

        if decisor_type is None:
            raise TypeError(f"{cls} must redefine '_dm_type'")

    def validate_data(self, **kwargs):
        """Validate all the data previously to send to the real algorithm."""
        pass


# =============================================================================
# NORMALIZER MIXIN
# =============================================================================


class NormalizerMixin:
    """Mixin class for all normalizers in scikit-criteria."""

    _dm_type = "normalizer"

    def normalize_data(self, **kwargs):  # noqa: D401
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

    def normalize(self, dm):
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
