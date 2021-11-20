#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The Module implements utilities to build a composite decision-maker."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import Counter

from .core import SKCMethodABC
from .utils import Bunch


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
    applied together while setting different parameters. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another dmaker or a transformer removed by setting it to
    `'passthrough'` or `None`.

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

    def __init__(self, steps):
        self._validate_steps(steps)
        self.steps = list(steps)

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self.steps)

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
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(self.steps[ind])
        elif isinstance(ind, int):
            return self.steps[ind][-1]
        elif isinstance(ind, str):
            return self.named_steps[ind]
        raise KeyError(ind)

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

    @property
    def named_steps(self):
        """Dictionary-like object, with the following attributes.

        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

        """
        return Bunch("steps", dict(self.steps))

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
# FUNCTIONS
# =============================================================================


def _name_steps(steps):
    """Generate names for steps."""
    # Based on sklearn.pipeline._name_estimators

    steps = list(reversed(steps))

    names = [type(step).__name__.lower() for step in steps]

    name_count = {k: v for k, v in Counter(names).items() if v > 1}

    named_steps = []
    for name, step in zip(names, steps):
        count = name_count.get(name, 0)
        if count:
            name_count[name] = count - 1
            name = f"{name}_{count}"

        named_steps.append((name, step))

    named_steps.reverse()

    return named_steps


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
        Returns a scikit-learn :class:`SKCPipeline` object.

    """
    named_steps = _name_steps(steps)
    return SKCPipeline(named_steps)
