#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022, 2023, 2024 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to calculate the difference between two objects."""

# =============================================================================
# IMPORTS
# =============================================================================

import abc
from dataclasses import dataclass, field

# =============================================================================
# CLASSES
# =============================================================================


class WithDiff(abc.ABC):
    """Mixin to add a difference attribute."""

    @abc.abstractmethod
    def diff(self, other):
        """Returns the difference between two objects."""
        raise NotImplementedError()


class _Missing(object):
    def __new__(cls, *args, **kwargs):
        """Creates a new instance of the class if it does not already exist, \
        or returns the existing instance.
        """
        if not hasattr(cls, "__instance"):
            instance = super().__new__(cls, *args, **kwargs)
            setattr(cls, "__instance", instance)
        return getattr(cls, "__instance")

    def __repr__(self):
        """A string representation of the object."""
        return "<MISSING>"


#: A singleton object used to represent missing values.
MISSING = _Missing()


@dataclass(frozen=True)
class _Difference:
    """Holds the difference between two objects."""

    left_type: type
    right_type: type
    members_diff: dict = field(default_factory=dict, init=False)

    @property
    def different_types(self):
        """Returns True if the left_type and right_type are different, and \
        False otherwise.

        """
        return self.left_type is not self.right_type

    @property
    def has_differences(self):
        """Checks if the object has any differences.

        True if there are any differences, False otherwise.

        """
        return self.different_types or bool(self.members_diff)

    def __repr__(self):
        """Return a string representation of the object."""
        diff_types = self.different_types
        members_diff = tuple(sorted(self.members_diff))
        return (
            f"<Difference different_types={diff_types!r} "
            f"members_diff={members_diff!r}>"
        )


def diff(left, right, **members):
    """Calculates the difference between two objects, `left` and `right`,
    and returns a `Difference` object.

    Parameters
    ----------
    left : object
        The first object to compare.
    right : object
        The second object to compare.
    **members : dict
        Keyword named arguments representing members to compare. The
        values of the members is the function to compare the members values

    Returns
    -------
    Difference
        A `Difference` object representing the differences between the two
        objects.

    Notes
    -----
    This function compares the values of corresponding members in the `left`
    and `right` objects. If a member is missing in either object, it is
    considered a difference. If a member is present in both objects, it is
    compared using the corresponding comparison function specified in
    `members`.

    Examples
    --------
    >>> obj_a = SomeClass(a=1, b=2)
    >>> obj_b = SomeClass(a=1, b=3, c=4)
    >>> diff(obj_a, obj_b, a=np.equal, b=np.equal)
    <Difference different_types=False members_diff=('b', 'c')>

    """
    # the difference container
    the_diff = _Difference(left_type=type(left), right_type=type(right))

    # if the objects are the same, no need to run the test
    # check if the types are different, if so, return the difference
    if left is right or the_diff.different_types:
        return the_diff

    # cheke one meber  by one
    for member, member_cmp in members.items():
        # retrieve the values
        lvalue = getattr(left, member, MISSING)
        rvalue = getattr(right, member, MISSING)

        # check if the member is missing in only one object
        if {lvalue is MISSING, rvalue is MISSING} == {True, False}:
            the_diff.members_diff[member] = (lvalue, rvalue)

        # check if members are different based on the provided
        # comparation function
        elif not member_cmp(lvalue, rvalue):
            the_diff.members_diff[member] = (lvalue, rvalue)

    return the_diff
