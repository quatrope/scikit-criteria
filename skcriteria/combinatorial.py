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

from collections import OrderedDict
import itertools

from .utils import hidden

with hidden():
    from .core import SKCMethodABC
    from .pipeline import SKCPipeline
    from .utils import Bunch, unique_names


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_step_compmbinations(steps):
    for name, step in steps:
        if not isinstance(step, list):
            step = [step]
        yield (name, step)


# =============================================================================
# CLASS
# =============================================================================


class CombinatorialPipeline(SKCMethodABC):

    _skcriteria_dm_type = "combinatorial_pipeline"
    _skcriteria_parameters = ["steps"]

    def __init__(self, steps):
        steps = list(steps)
        self._validate_steps(steps)
        self._steps = steps
        self._pipelines = OrderedDict()

        all_combinations = itertools.product(*_get_step_compmbinations(steps))
        for idx, combo in all_combinations:
            pipeline = SKCPipeline(combo)
            pipeline_name = "-".join([name for name, _ in combo])
            self._pipelines[pipeline_name] = pipeline

    def _validate_input_steps(self, steps):
        if not isinstance(steps, list):
            raise TypeError("Steps must be a list.")
        if not steps:
            raise ValueError("Steps cannot be empty.")

        # Basic validation: ensure all elements are components or
        # lists of components
        for , step_group in enumerate(steps):
            if isinstance(step_group, list):
                if not step_group:
                    raise ValueError(
                        f"Empty list of alternatives at step {i}."
                    )
                for component in step_group:
                    if not (
                        hasattr(component, "transform")
                        or hasattr(component, "evaluate")
                    ):
                        raise TypeError(
                            f"Component {component} in alternative list at step {i} "
                            "does not implement 'transform()' or 'evaluate()'."
                        )
            else:
                if not (
                    hasattr(step_group, "transform")
                    or hasattr(step_group, "evaluate")
                ):
                    raise TypeError(
                        f"Component {step_group} at step {i} "
                        "does not implement 'transform()' or 'evaluate()'."
                    )

    @property
    def _steps(self):
        """The raw steps provided during initialization, including alternatives."""
        return self._steps

    def evaluate(self, dm):
        """Evaluates all generated pipelines with the given DecisionMatrix.

        Parameters
        ----------
        dm : :py:class:`skcriteria.core.DecisionMatrix`
            The decision matrix to evaluate.

        Returns
        -------
        results : dict
            A dictionary where keys are pipeline names (e.g., "normalizer-topsis")
            and values are the results from each `SKCPipeline.evaluate()` call.
        """

    def evaluate(self, dm):
        """Evaluates all generated pipelines with the given DecisionMatrix."""
        results = Bunch("combinatorial_results")
        for pipeline_name, pipeline in self._iter_pipelines():
            try:
                results[pipeline_name] = pipeline.evaluate(dm)
            except TypeError as e:
                # Catch errors from SKCPipeline validation if the last step is not a DM
                print(
                    f"Skipping pipeline {pipeline_name} due to validation error: {e}"
                )
                continue
            except Exception as e:
                print(f"Error evaluating pipeline {pipeline_name}: {e}")
                continue

        return results
