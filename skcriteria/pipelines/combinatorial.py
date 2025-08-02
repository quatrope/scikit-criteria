#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""The Module implements utilities to build a combinatorial pipeline."""

# =============================================================================
# IMPORTS
# =============================================================================

import itertools

from .simple_pipeline import SKCPipeline
from ..cmp import RanksComparator
from ..core import SKCMethodABC
from ..utils import Bunch, unique_names

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _make_all_combinations_pipelines(steps):
    names, steps_groups = [], []
    for name, step_group in steps:
        if not isinstance(step_group, list):
            step_group = [step_group]
        names.append(name)
        steps_groups.append(step_group)

    steps_combs = itertools.product(*steps_groups)

    pipelines_names, pipelines = [], []
    for comb in steps_combs:
        pipeline_name = "_".join([step.get_method_name() for step in comb])
        pipeline_steps = list(zip(names, comb))

        pipeline = SKCPipeline(steps=pipeline_steps)

        pipelines_names.append(pipeline_name)
        pipelines.append(pipeline)

    return unique_names(names=pipelines_names, elements=pipelines)


# =============================================================================
# CLASS
# =============================================================================


class SKCCombinatorialPipeline(SKCMethodABC):
    """Model that encapsulates a pipeline of MCDA methods with alternatives.

    This class allows you to define a sequential pipeline of data
    transformation and aggregation steps, where some steps may have multiple
    alternative implementations. The ``CombinatorialPipeline`` will generate
    all possible pipelines by combining these alternatives and evaluate them.

    Parameters
    ----------
    steps : list of (str, method or list of methods)
        List of (name, transform) tuples (implementing ``fit/transform``) that
        are chained, in the order in which they are chained. Steps can be a
        single method or a list of alternative methods.

        .. code-block:: python

            # simple pipeline
            steps = [
                ("inverter", invert_objectives.InvertObjectives()),
                ("scaler", scalers.SumScaler(target="matrix")),
                ("agg", simple.WeightedSum())
            ]

            # pipeline with alternatives in the scaler step
            steps = [
                ("inverter", invert_objectives.InvertObjectives()),
                (
                    "scaler",
                    [
                        scalers.SumScaler(target="matrix"),
                        scalers.VectorScaler(target="matrix"),
                    ],
                ),
                ("agg", simple.WeightedSum()),
            ]

    """

    _skcriteria_dm_type = "combinatorial_pipeline"
    _skcriteria_parameters = ["steps"]

    def __init__(self, steps):
        steps = list(steps)
        if len(steps) < 2:
            raise ValueError("Pipeline must have at least two steps.")

        self._steps = steps
        self._pipelines = _make_all_combinations_pipelines(steps)

    @property
    def steps(self):
        """The raw steps provided during initialization."""
        return list(self._steps)

    @property
    def named_steps(self):
        """The raw steps provided during initialization as a dict-like."""
        return Bunch("steps", dict(self.steps))

    @property
    def pipelines(self):
        """List of all generated pipelines."""
        return list(self._pipelines)

    @property
    def named_pipelines(self):
        """A dict-like of all generated pipelines."""
        return Bunch("pipelines", dict(self.pipelines))

    def __len__(self):
        """Return the length of the Pipeline (the sum of all pipelines)."""
        return sum(map(len, self.named_pipelines.values()))

    def transform(self, dm):
        """Transform the data, without applying the final evaluator.

        This method applies the transformation steps (such as inverters and
        scalers) of each individual pipeline to the input decision matrix.
        It does not execute the final aggregation step (the evaluator),
        behaving analogously to the `transform` method of a standard
        `SKCPipeline`.

        Since multiple pipelines are generated, this method does not return a
        single transformed decision matrix, but rather a dictionary-like
        containing all transformed matrices, each associated with the unique
        name of the pipeline that generated it.

        Parameters
        ----------
        dm : :py:class:`skcriteria.core.DecisionMatrix`
            The decision matrix to transform.

        Returns
        -------
        dict-like
            A dictionary mapping the name of each pipeline (str) to its
            corresponding transformed
            :py:class:`skcriteria.core.DecisionMatrix`.

        See Also
        --------
        SKCCombinatorialPipeline.evaluate : Evaluate all pipelines and
            compare their final rankings.
        SKCCombinatorialPipeline.pipelines : Access the list of individual
            generated pipelines.
        SKCPipeline.transform : Transforms the data using a single
            pipeline.

        """
        dmts = []
        for pipeline_name, pipeline in self._pipelines:
            dmts.append((pipeline_name, pipeline.transform(dm)))
        return Bunch("transformed_dm", dict(dmts))

    def evaluate(self, dm):
        """Evaluates all generated pipelines with the given DecisionMatrix.

        Parameters
        ----------
        dm : :py:class:`skcriteria.core.DecisionMatrix`
            The decision matrix to evaluate.

        Returns
        -------
        :py:class:`skcriteria.cmp.RanksComparator`
            A comparator object containing the ranks of all alternatives for
            each generated pipeline.

        """
        ranks = []
        for pipeline_name, pipeline in self._pipelines:
            ranks.append((pipeline_name, pipeline.evaluate(dm)))

        return RanksComparator(ranks, {})


# =============================================================================
# FACTORY
# =============================================================================


def mkcombinatorial(*steps):
    """Construct a CombinatorialPipeline from the given transformers and \
    decision-maker.

    This is a shorthand for the CombinatorialPipeline constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps: list of transformers and decision-maker object
        List of the scikit-criteria transformers and decision-maker
        that are chained together.

    Returns
    -------
    p : CombinatorialPipeline
        Returns a scikit-criteria :class:`CombinatorialPipeline` object.

    """
    names = []
    for step in steps:
        if isinstance(step, list):
            name = "_".join([type(s).__name__.lower() for s in step])
        else:
            name = type(step).__name__.lower()
        names.append(name)

    named_steps = unique_names(names=names, elements=steps)
    return SKCCombinatorialPipeline(named_steps)
