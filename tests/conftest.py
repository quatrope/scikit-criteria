#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.data"""


# =============================================================================
# IMPORTS
# =============================================================================

import functools
import inspect
import os
from pathlib import Path

import matplotlib as mpl

import numpy as np

import pytest

from skcriteria import core


# =============================================================================
# CONSTANTS
# =============================================================================


MAXS_O_ALIAS = list(core.Objective._MAX_ALIASES.value)
MINS_O_ALIAS = list(core.Objective._MIN_ALIASES.value)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def data_values():
    def make(
        *,
        seed=None,
        min_alternatives=3,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=10,
        min_objectives_proportion=0.5,
        nan_proportion=0,
    ):
        # start the random generator
        random = np.random.default_rng(seed=seed)

        # determine the number os alternatives and criteria
        alternatives_number = (
            random.integers(min_alternatives, max_alternatives)
            if min_alternatives != max_alternatives
            else min_alternatives
        )
        criteria_number = (
            random.integers(min_criteria, max_criteria)
            if min_criteria != max_criteria
            else min_criteria
        )

        # create the data matrix with rows = alt and columns = crit
        mtx = random.random((alternatives_number, criteria_number))

        # if we have a nan ratio >0 of nan we have to add them randomly
        # in the matrix
        if nan_proportion:
            nan_number = round(mtx.size * float(nan_proportion))
            nan_positions = random.choice(mtx.size, nan_number, replace=False)
            mtx.ravel()[nan_positions] = np.nan

        # determine the number of minimize objectives based on the proportion
        # of the total number of criteria, and the maximize is the complement
        min_objectives_number = round(
            criteria_number * min_objectives_proportion
        )
        max_objectives_number = round(
            criteria_number * 1.0 - min_objectives_number
        )

        # if the proportion is lt or gt than the total number of criteria
        # we add or remove an objective
        while min_objectives_number + max_objectives_number < criteria_number:
            if random.choice([True, False]):
                max_objectives_number += 1
            else:
                min_objectives_number += 1
        while min_objectives_number + max_objectives_number > criteria_number:
            if random.choice([True, False]):
                max_objectives_number -= 1
            else:
                min_objectives_number -= 1

        # finally we extract the objectives and shuffle the order
        objectives = np.concatenate(
            [
                random.choice(MINS_O_ALIAS, min_objectives_number),
                random.choice(MAXS_O_ALIAS, max_objectives_number),
            ]
        )
        random.shuffle(objectives)

        # the weights
        weights = random.random(criteria_number)

        # create the names of the criteria and the alternatives
        alternatives = [f"A{idx}" for idx in range(alternatives_number)]
        criteria = [f"C{idx}" for idx in range(criteria_number)]

        return mtx, objectives, weights, alternatives, criteria

    return make


@pytest.fixture(scope="session")
def decision_matrix(data_values):
    @functools.wraps(data_values)
    def make(**kwargs):
        mtx, objectives, weights, alternatives, criteria = data_values(
            **kwargs
        )

        dm = core.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            alternatives=alternatives,
            criteria=criteria,
        )

        return dm

    return make


# =============================================================================
# MARKERS
# =============================================================================


MARKERS = [
    ("posix", "os.name != 'posix'", "Requires a POSIX os"),
    ("not_posix", "os.name == 'posix'", "Skipped on POSIX"),
    ("windows", "os.name != 'nt'", "Requires Windows"),
    ("not_windows", "os.name == 'nt'", "Skipped on Windows"),
    ("linux", "not sys.platform.startswith('linux')", "Requires Linux"),
    ("not_linux", "sys.platform.startswith('linux')", "Skipped on Linux"),
    ("osx", "sys.platform != 'darwin'", "Requires OS X"),
    ("not_osx", "sys.platform == 'darwin'", "Skipped on OS X"),
]


def pytest_configure(config):
    for marker, _, reason in MARKERS:
        config.addinivalue_line("markers", "{}: {}".format(marker, reason))


def pytest_collection_modifyitems(items):
    for item in items:
        for searched_marker, condition, default_reason in MARKERS:
            marker = item.get_closest_marker(searched_marker)
            if not marker:
                continue

            if "reason" in marker.kwargs:
                reason = "{}: {}".format(
                    default_reason, marker.kwargs["reason"]
                )
            else:
                reason = default_reason + "."
            skipif_marker = pytest.mark.skipif(condition, reason=reason)
            item.add_marker(skipif_marker)


# =============================================================================
# CI
# =============================================================================


mpl.use("Agg")


def _patched_image_directories(func):
    """Patch _image_directories so tests don't collide on tox run-parallel

    Matplotlib's check_figures_equal decorator uses this to determine where to
    save images.
    """
    wdir = os.environ.get("TOX_ENV_DIR", ".")  # current dir if no tox
    module_path = Path(inspect.getfile(func))
    baseline_dir = module_path.parent / "baseline_images" / module_path.stem
    result_dir = Path(wdir).resolve() / "result_images" / module_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    print(baseline_dir, result_dir)
    return baseline_dir, result_dir


pytest.MonkeyPatch().setattr(
    "matplotlib.testing.decorators._image_directories",
    _patched_image_directories,
)
