#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""test for skcriteria.data

"""


# =============================================================================
# IMPORTS
# =============================================================================

import functools


import numpy as np

import pytest

import skcriteria


# =============================================================================
# CONSTANTS
# =============================================================================


MAXS_O_ALIAS = list(skcriteria.Objective._MAX_ALIASES.value)
MINS_O_ALIAS = list(skcriteria.Objective._MIN_ALIASES.value)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def data_values():
    def make(
        seed=None,
        min_alternatives=3,
        max_alternatives=10,
        min_criteria=3,
        max_criteria=10,
        min_objectives_proportion=0.5,
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

        # determine the number of minimize objectives bases on the proportion
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
        anames = [f"A{idx}" for idx in range(alternatives_number)]
        cnames = [f"C{idx}" for idx in range(criteria_number)]

        return mtx, objectives, weights, anames, cnames

    return make


@pytest.fixture(scope="session")
def decision_matrix(data_values):
    @functools.wraps(data_values)
    def make(*args, **kwargs):
        mtx, objectives, weights, anames, cnames = data_values(*args, **kwargs)

        dm = skcriteria.mkdm(
            matrix=mtx,
            objectives=objectives,
            weights=weights,
            anames=anames,
            cnames=cnames,
        )

        return dm

    return make
