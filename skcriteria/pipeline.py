#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The Module implements utilities to build a composite decision-maker."""

# =============================================================================
# IMPORTS
# =============================================================================

from .core import SKCMethodABC
from .utils import Bunch, unique_names


# =============================================================================
# CLASS
# =============================================================================
class SKCPipeline(SKCMethodABC):
    """Pipeline of transforms with a final decision-maker.

    Sequentially apply a list of transforms and a final decisionmaker.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement `transform` method.

    The final decision-maker only needs to implement `evaluate`.

    The purpose of the pipeline is to assemble several steps that can be
    applied together while setting different parameters.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing evaluate/transform)
        that are chained, in the order in which they are chained, with the last
        object an decision-maker.

    See Also
    --------
    skcriteria.pipeline.mkpipe : Convenience function for simplified
        pipeline construction.

    """

    _skcriteria_dm_type = "pipeline"
    _skcriteria_parameters = ["steps"]

    def __init__(self, steps):
        steps = list(steps)
        self._validate_steps(steps)
        self._steps = steps

    # INTERNALS ===============================================================

    def _validate_steps(self, steps):
        for name, step in steps[:-1]:
            if not isinstance(name, str):
                raise TypeError("step names must be instance of str")
            if not (hasattr(step, "transform") and callable(step.transform)):
                raise TypeError(
                    f"step '{name}' must implement 'transform()' method"
                )

        name, dmaker = steps[-1]
        if not isinstance(name, str):
            raise TypeError("step names must be instance of str")
        if not (hasattr(dmaker, "evaluate") and callable(dmaker.evaluate)):
            raise TypeError(
                f"step '{name}' must implement 'evaluate()' method"
            )

    # PROPERTIES ==============================================================

    @property
    def steps(self):
        """List of steps of the pipeline."""
        return list(self._steps)

    @property
    def named_steps(self):
        """Dictionary-like object, with the following attributes.

        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

        """
        return Bunch("steps", dict(self.steps))

    # DUNDERS =================================================================

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self._steps)

    def __getitem__(self, ind):
        """Return a sub-pipeline or a single step in the pipeline.

        Indexing with an integer will return an step; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying steps in the sub-pipeline
        will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.

        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                cname = type(self).__name__
                raise ValueError(f"{cname} slicing only supports a step of 1")
            return self.__class__(self.steps[ind])
        elif isinstance(ind, int):
            return self.steps[ind][-1]
        elif isinstance(ind, str):
            return self.named_steps[ind]
        raise KeyError(ind)

    # API =====================================================================

    def evaluate(self, dm):
        """Run the all the transformers and the decision maker.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the result will be calculated.

        Returns
        -------
        r : Result
            Whatever the last step (decision maker) returns from their evaluate
            method.

        """
        dm = self.transform(dm)
        _, dmaker = self.steps[-1]
        result = dmaker.evaluate(dm)
        return result

    def transform(self, dm):
        """Run the all the transformers.

        Parameters
        ----------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Decision matrix on which the transformations will be applied.

        Returns
        -------
        dm: :py:class:`skcriteria.data.DecisionMatrix`
            Transformed decision matrix.

        """
        for _, step in self.steps[:-1]:
            dm = step.transform(dm)
        return dm


# =============================================================================
# FACTORY
# =============================================================================


def mkpipe(*steps):
    """Construct a Pipeline from the given transformers and decision-maker.

    This is a shorthand for the SKCPipeline constructor; it does not require,
    and does not permit, naming the estimators. Instead, their names will
    be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps: list of transformers and decision-maker object
        List of the scikit-criteria transformers and decision-maker
        that are chained together.

    Returns
    -------
    p : SKCPipeline
        Returns a scikit-criteria :class:`SKCPipeline` object.

    """
    names = [type(step).__name__.lower() for step in steps]
    named_steps = unique_names(names=names, elements=steps)
    return SKCPipeline(named_steps)
