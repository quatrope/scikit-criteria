#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to calculate the difference between two objects."""

# =============================================================================
# IMPORTS
# =============================================================================

import abc
import inspect
from dataclasses import dataclass, field

# =============================================================================
# The utilities classes
# =============================================================================


class _Missing(object):
    """A singleton object used to represent missing values."""

    def __new__(cls, *args, **kwargs):
        """Creates a new instance of the class if it does not already exist, \
        or returns the existing instance."""
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
        False otherwise."""
        return self.left_type is not self.right_type

    @property
    def has_differences(self):
        """Checks if the object has any differences.

        True if there are any differences, False otherwise.

        """
        return self.different_types or bool(self.members_diff)

    def __repr__(self):
        """Return a string representation of the object."""
        has_differences = self.has_differences
        diff_types = self.different_types
        members_diff = tuple(sorted(self.members_diff))
        return (
            f"<Difference has_differences={has_differences!r} "
            f"different_types={diff_types!r} "
            f"members_diff={members_diff!r}>"
        )


def diff(left, right, **members):
    """Calculates the difference between two objects, `left` and `right`, \
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


# =============================================================================
# The mixin classes to implement diff basses equalities
# =============================================================================


class DiffEqualityMixin(abc.ABC):
    """Abstract base class for classes with a diff method.

    This class provides methods for comparing objects with a tolerance,
    allowing for differences within specified limits. It is designed to be
    used with numpy and pandas equality functions.

    Extra methods:

    - ``aequals``
        almost-equals, Check if the two objects are equal within a tolerance.
    - ``equals(other)``
        Return True if the objects are equal.
    - ``__eq__(other)``
        Implement equality comparison.
    - ``__ne__(other)``
        Implement inequality comparison.

    """

    def __init_subclass__(cls):
        """Validate the creation of a subclass."""
        o_params = list(inspect.signature(DiffEqualityMixin.diff).parameters)
        params = list(inspect.signature(cls.diff).parameters)
        if o_params != params:
            o_params.remove("self")
            diff_method_name = cls.diff.__qualname__
            raise TypeError(
                f"{diff_method_name!r} must redefine exactly "
                f"the parameters {o_params!r}"
            )

    @abc.abstractmethod
    def diff(
        self, other, rtol=1e-05, atol=1e-08, equal_nan=True, check_dtypes=False
    ):
        """Return the difference between two objects within a tolerance.

        This method should be implemented by subclasses to define how
        differences between objects are calculated.

        The tolerance parameters rtol and atol, equal_nan, and check_dtypes are
        provided to be used by the numpy and pandas equality functions.
        These parameters allow you to customize the behavior of the equality
        comparison, such as setting the relative and absolute tolerance for
        numeric comparisons, considering NaN values as equal, and checking
        for the data type of the objects being compared.

        Notes
        -----
        The tolerance values are positive, typically very small numbers.  The
        relative difference (`rtol` * abs(`b`)) and the absolute difference
        `atol` are added together to compare against the absolute difference
        between `a` and `b`.

        NaNs are treated as equal if they are in the same place and if
        ``equal_nan=True``.  Infs are treated as equal if they are in the same
        place and of the same sign in both arrays.

        Parameters
        ----------
        other : object
            The object to compare to.
        rtol : float, optional
            The relative tolerance parameter. Default is 1e-05.
        atol : float, optional
            The absolute tolerance parameter. Default is 1e-08.
        equal_nan : bool, optional
            Whether to consider NaN values as equal. Default is True.
        check_dtypes : bool, optional
            Whether to check the data type of the objects. Default is False.

        Returns
        -------
        the_diff :
            The difference between the current and the other object.

        See Also
        --------
        equals, aequals, :py:func:`numpy.isclose`, :py:func:`numpy.all`,
        :py:func:`numpy.any`, :py:func:`numpy.equal`,
        :py:func:`numpy.allclose`.

        """
        raise NotImplementedError()

    def aequals(
        self,
        other,
        *,
        rtol=1e-05,
        atol=1e-08,
        equal_nan=True,
        check_dtypes=False,
    ):
        """Check if the two objects are equal within a tolerance.

        All the parameters ara passed to the `diff` method.


        Parameters
        ----------
        other : object
            The object to compare to.
        rtol : float, optional
            The relative tolerance parameter. Default is 1e-05.
        atol : float, optional
            The absolute tolerance parameter. Default is 1e-08.
        equal_nan : bool, optional
            Whether to consider NaN values as equal. Default is True.
        check_dtypes : bool, optional
            Whether to check the data type of the objects. Default is False.

        Returns
        -------
        bool
            True if the objects are equal within the specified tolerance,
            False otherwise.

        """
        the_diff = self.diff(
            other,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            check_dtypes=check_dtypes,
        )
        is_aequals = not the_diff.has_differences
        return is_aequals

    def equals(self, other):
        """Return True if the objects are equal.

        This method calls `aquals()` without tolerance.

        Parameters
        ----------
        other : :py:class:`object`
            Other instance to compare.

        Returns
        -------
        equals : :py:class:`bool:py:class:`
            Returns True if the two objects are equals.

        See Also
        --------
        aequals, diff.

        """
        return self.aequals(
            other, rtol=0, atol=0, equal_nan=False, check_dtypes=True
        )

    def __eq__(self, other):
        """x.__eq__(y) <==> (x == y) <==> x.equals(y)."""
        return self.equals(other)

    def __ne__(self, other):
        """x.__ne__(y) <==> (x != y) <==> not x.equals(y)."""
        return not self == other
