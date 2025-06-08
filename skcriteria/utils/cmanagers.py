#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Multiple context managers to use inside scikit-criteria."""

# =============================================================================
# IMPORTS
# =============================================================================

import contextlib
import inspect

# =============================================================================
# TEMPORAL HEADER
# =============================================================================


@contextlib.contextmanager
def df_temporal_header(df, header, name=None):
    """Temporarily replaces a DataFrame columns names.

    Optionally also assign another name to the columns.

    Parameters
    ----------
    header : sequence
        The new names of the columns.
    name : str or None (default None)
        New name for the index containing the columns in the DataFrame. If
        'None' the original name of the columns present in the DataFrame is
        preserved.

    """
    original_header = df.columns
    original_name = original_header.name

    name = original_name if name is None else name
    try:
        df.columns = header
        df.columns.name = name
        yield df
    finally:
        df.columns = original_header
        df.columns.name = original_name


# =============================================================================
# EHIDDEN GLOBAL ATTRIBUTES
# =============================================================================


class HiddenAlreadyUsedInThisContext(RuntimeError):
    """Raised when a context attempts to use the 'hidden' context manager \
    more than once within the same scope."""


class NonGlobalHidden(RuntimeError):
    """Exception raised when the 'hidden' decorator is used in a context that \
    is not the global scope of a module.

    This exception indicates that the 'hidden' decorator should only be
    applied globally, outside of any functions or methods, and an attempt to
    use it within a local context (e.g., inside a function or method) has
    been detected.

    """


class _DirWithHidden:
    """Custom directory function with hidden objects filtering.

    Parameters
    ----------
    frame : frame
        The frame whose global variables will be considered.
    hidden_objects : dict
        A dictionary containing names of objects to be hidden and their
        corresponding objects.

    Examples
    --------
    >>> frame = inspect.currentframe()
    >>> hidden = {'obj_to_hide': object()}
    >>> dir_with_hidden = _DirWithHidden(frame, hidden)
    >>> visible_attrs = dir_with_hidden()
    >>> print(visible_attrs)
    ['visible_obj1', 'visible_obj2']

    """

    def __init__(self, frame, hidden_objects):
        self.frame = frame
        self.hidden_objects = hidden_objects

    def __call__(self):
        """Call method to retrieve visible attributes in the frame.

        Returns
        -------
        list
            A list of visible attribute names in the frame, considering
            the hidden_objects.

        """
        attrs = []
        for obj_name, obj in self.frame.f_globals.items():
            if obj_name not in self.hidden_objects:
                attrs.append(obj_name)
            elif self.hidden_objects[obj_name] is not obj:
                attrs.append(obj_name)
        return attrs


@contextlib.contextmanager
def hidden(*, hide_this=True, dry=False):
    """A context manager for hiding objects in the global scope.

    Parameters
    ----------
    hide_this : bool, optional
        Whether to hide the 'hidden' context manager itself and/or the hidden
        module. Defaults to True.
    dry : bool, optional, default False
        If is True, the objects are not hide. Useful for testing.


    Raises
    ------
    NonGlobalHidden
        If 'hidden' is declared inside a function, class or method.

    HiddenAlreadyUsedInThisContext
        If the 'hidden' context manager is used more than once in the same
        context.


    Yields
    ------
    None


    Notes
    -----
    - This context manager is intended to be used globally (outside any
      functions or methods).
    - It hides objects within the global scope for the duration of the context.

    Implementation Details
    ----------------------
    - The context manager retrieves the current frame and ensures it is used
      globally.
    - It captures the state of the global scope before entering the context.
    - Objects introduced within the context are hidden in the global scope.
    - The '__dir__' attribute of the global scope is customized to include
      logic to hide the objects introduced within the context.

    """
    self = hidden  # this function

    # two levels because of the decorator @contextmanager
    frame = inspect.currentframe().f_back.f_back

    # if co_name != <module> we are inside a function or class
    if frame.f_code.co_name != "<module>":
        check_here = (
            f"{frame.f_code.co_filename}:"
            f"{frame.f_lineno}::"
            f"{frame.f_code.co_name}"
        )
        raise NonGlobalHidden(
            f"hidden() can only be used globally. Check {check_here!r}"
        )

    # get the current state
    pre_f_globals = dict(frame.f_globals)

    # if he current state alreade replace the __dir__ hidden is used two times
    if isinstance(pre_f_globals.get("__dir__"), _DirWithHidden):
        raise HiddenAlreadyUsedInThisContext(frame.f_code.co_filename)

    yield  # execute the decorator

    # check whatever the code declared inside the context
    hidden_objects = {}
    for obj_name, obj in frame.f_globals.items():
        if obj_name not in pre_f_globals:
            hidden_objects[obj_name] = obj
        elif obj is self and hide_this:
            hidden_objects[obj_name] = obj

    # create the new dir object
    custom_dir = _DirWithHidden(frame=frame, hidden_objects=hidden_objects)

    # if dry, don't do anything simply stop now
    # otherwise replace the global __dir__
    if not dry:
        frame.f_globals["__dir__"] = custom_dir
