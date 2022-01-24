#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Framework for defining models with hyperparameters."""


# =============================================================================
# IMPORTS
# =============================================================================

from abc import ABCMeta

import attr


# =============================================================================
# CONSTANTS
# =============================================================================

HMODEL_METADATA_KEY = "__hmodel__"

HMODEL_CONFIG = "__hmodel_config__"

DEFAULT_HMODEL_CONFIG = {"repr": False, "frozen": False}


# =============================================================================
# FUNCTIONS
# =============================================================================


def hparam(default, **kwargs):
    """Create a hyper parameter.

    By design decision, hyper-parameter is required to have a sensitive default
    value.

    Parameters
    ----------
    default :
        Sensitive default value of the hyper-parameter.
    **kwargs :
        Additional keyword arguments are passed and are documented in
        ``attr.ib()``.

    Return
    ------
    Hyper parameter with a default value.

    Notes
    -----
    This function is a thin-wrapper over the attrs function ``attr.ib()``.
    """
    field_metadata = kwargs.pop("metadata", {})
    model_metadata = field_metadata.setdefault(HMODEL_METADATA_KEY, {})
    model_metadata["hparam"] = True
    return attr.field(
        default=default, metadata=field_metadata, kw_only=True, **kwargs
    )


def mproperty(**kwargs):
    """Create a internal property for the model.

    By design decision, hyper-parameter is required to have a sensitive default
    value.

    Parameters
    ----------
    default :
        Sensitive default value of the hyper-parameter.
    **kwargs :
        Additional keyword arguments are passed and are documented in
        ``attr.ib()``.

    Return
    ------
    Hyper parameter with a default value.

    Notes
    -----
    This function is a thin-wrapper over the attrs function ``attr.ib()``.

    """
    field_metadata = kwargs.pop("metadata", {})
    model_metadata = field_metadata.setdefault(HMODEL_METADATA_KEY, {})
    model_metadata["mproperty"] = True
    return attr.field(init=False, metadata=field_metadata, **kwargs)


@attr.define(repr=False)
class HModelABC(metaclass=ABCMeta):
    """Create a new models with hyperparameters."""

    def __init_subclass__(cls):
        """Initiate of subclasses.

        It ensures that every inherited class is decorated by ``attr.define()``
        and assigns as class configuration the parameters defined in the class
        variable `__hmodel_config__`.

        In other words it is slightly equivalent to:

        .. code-block:: python

            @attr.s(**HModelABC.__hmodel_config__)
            class Decomposer(HModelABC):
                pass

        """
        model_config = getattr(cls, HMODEL_CONFIG, DEFAULT_HMODEL_CONFIG)
        return attr.define(maybe_cls=cls, slots=False, **model_config)

    @classmethod
    def get_hparams(cls):
        """Return a tuple of available hyper parameters."""

        def flt(f):
            return f.metadata[HMODEL_METADATA_KEY].get("hparam", False)

        return tuple(f.name for f in attr.fields(cls) if flt(f))

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        clsname = type(self).__name__

        hparams = self.get_hparams()
        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: attr.name in hparams and attr.repr,
        )

        attrs_str = ", ".join(
            [f"{k}={repr(v)}" for k, v in sorted(selfd.items())]
        )
        return f"{clsname}({attrs_str})"
