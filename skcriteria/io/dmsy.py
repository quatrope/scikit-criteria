#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""DMSY (Decision Matrix Simple YAML) format support.

This module provides functions to read and write DecisionMatrix objects
in DMSY format, a simple YAML-based format designed for easy human
readability and editing of multi-criteria decision analysis data.

The DMSY format generates YAML files optimized for human readability
through intelligent use of flow and block styles. One-dimensional arrays are
represented in compact flow style (e.g., [1, 2, 3]), while multi-dimensional
structures use block style for the outer dimension with flow style for inner
dimensions (e.g., - [1, 2, 3] on separate lines). This combination allows
complex decision matrices to maintain a clear and navigable structure,
facilitating both manual inspection and direct file editing, without
sacrificing compactness when appropriate.

"""

# =============================================================================
# Imports
# =============================================================================

import datetime as dt
import importlib.metadata
import pathlib
import platform
import sys

import numpy as np

import yaml

from ..core import mkdm


# =============================================================================
# Constants
# =============================================================================

#: Default decision matrix type for DMSY format.
#: Currently only supports discrete multi-criteria decision matrices.
DEFAULT_DM_TYPE = "discrete-dm"

#: Default DMSY format version number.
#: Version 1 is the current stable format specification.
DEFAULT_DMSY_VERSION = 1

#: Template for DMSY file metadata containing software and environment
#: information.
#:
#: This dictionary contains default metadata fields that are automatically
#: included in DMSY files when saving DecisionMatrix objects. The metadata
#: provides information about the software version, authors, platform, and
#: creation context.
_DMSY_METADATA_DEFAULT_TEMPLATE = {
    "description": "Decision Matrix Simple YAML format",
    "skcriteria": importlib.metadata.version("scikit-criteria"),
    "authors": "Cabral, Luczywo & QuatroPe",
    "author_email": "jbcabral@unc.edu.ar",
    "url": "https://github.com/quatrope/scikit-criteria",
    "documentation": "https://scikit-criteria.quatrope.org/",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "python_version": sys.version,
    "created_at": None,
}


# =============================================================================
# YAML DUMPER
# =============================================================================

#: Mapping from NumPy dtype kind characters to Python native types.
#:
#: This dictionary maps NumPy's single-character dtype kind codes to their
#: corresponding Python native types for YAML serialization. Used by the
#: numpy_array_representer to convert NumPy arrays to Python lists with
#: appropriate types.
#:
#: Keys
#: ----
#: "f" : float
#:     Floating point types (float16, float32, float64, etc.)
#: "i" : int
#:     Signed integer types (int8, int16, int32, int64, etc.)
#: "u" : int
#:     Unsigned integer types (uint8, uint16, uint32, uint64, etc.)
#: "b" : bool
#:     Boolean type (bool_)
_NUMPY_TO_PYTHON_DTYPE_MAP = {
    "f": float,  # floating
    "i": int,  # signed integer
    "u": int,  # unsigned integer
    "b": bool,  # boolean
}


# this dumper is made public for easy testing
class CustomYAMLDumper(yaml.SafeDumper):
    """Custom YAML Dumper that handles numpy arrays and uses flow style \
    for lists.

    This dumper automatically converts numpy arrays to Python lists and formats
    all lists using flow style (e.g., [1, 2, 3] instead of block style).

    """


def _numpy_array_representer(dumper, obj):
    """Convert numpy arrays to Python lists with smart flow style.

    Uses np.ndim() to determine flow style: arrays with ndim >= 2 use block
    style for the outer dimension, while 1D arrays use flow style throughout.

    Parameters
    ----------
    dumper : yaml.SafeDumper
        The YAML dumper instance.
    obj : numpy.ndarray
        The numpy array to be converted.

    Returns
    -------
    yaml.SequenceNode
        YAML sequence node representing the array as a list with appropriate
        flow_style.

    Examples
    --------
    1D array (flow style):
        [1.0, 2.0, 3.0]

    2D+ array (block style for outer, flow for inner):
        - [1.0, 2.0, 3.0]
        - [4.0, 5.0, 6.0]
    """
    target_type = _NUMPY_TO_PYTHON_DTYPE_MAP.get(obj.dtype.kind)

    if target_type:
        obj = obj.astype(target_type)

    # Convert to Python list
    python_list = obj.tolist()

    # Use np.ndim() to determine flow style consistently
    dimensions = np.ndim(obj)
    flow_style = dimensions < 2

    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        python_list,
        flow_style=flow_style,
    )


def _numpy_scalar_representer(dumper, data):
    """Represent numpy scalars as Python native types.

    Parameters
    ----------
    dumper : yaml.SafeDumper
        The YAML dumper instance.
    data : numpy.generic
        The numpy scalar to be converted.

    Returns
    -------
    yaml.Node
        YAML node representing the scalar value.
    """
    item = data.item()
    if isinstance(item, np.generic):  # longdouble fix
        target_type = _NUMPY_TO_PYTHON_DTYPE_MAP.get(item.dtype.kind)
        item = target_type(item)

    return dumper.represent_data(item)


def _iterable_not_ndarray_representer(dumper, data):
    """Represent iterables (list,tuple, set, frozenset) as lists with smart \
    flow style.

    Uses np.ndim() to determine flow style consistently with other
    epresenters.
    Multi-dimensional structures use block style, 1D structures use flow style.

    Parameters
    ----------
    dumper : yaml.SafeDumper
        The YAML dumper instance.
    data : iterable
        The iterable to be converted to a list.

    Returns
    -------
    yaml.SequenceNode
        YAML sequence node with appropriate flow_style.
    """
    python_list = list(data)

    # Use np.ndim() for consistent dimensionality checking
    dimensions = np.ndim(python_list)
    flow_style = dimensions < 2

    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", python_list, flow_style=flow_style
    )


# Register representers in the custom dumper
CustomYAMLDumper.add_representer(np.ndarray, _numpy_array_representer)
CustomYAMLDumper.add_multi_representer(np.generic, _numpy_scalar_representer)
for cls in [list, tuple, set, frozenset]:
    CustomYAMLDumper.add_representer(cls, _iterable_not_ndarray_representer)

del cls  # cleanup

# =============================================================================
# DMSY HANDLERS
# =============================================================================

#: Registry of DMSY handlers by (dm_type, version) tuple.
#:
#: This dictionary maintains a mapping of available DMSY format handlers
#: for different decision matrix types and format versions. Handlers are
#: automatically registered using the @_register_dmsy_reader decorator.
#:
#: Keys are tuples of (dm_type, version) where:
#: - dm_type : str
#:     The type of decision matrix (e.g., "discrete-dm")
#: - version : int
#:     The DMSY format version number
#:
#: Values are handler classes that implement to_dm() and to_yml() methods.
_HANDLERS = {}


def _register_dmsy_reader(reader_cls):
    """Register a DMSY reader class for a specific dm_type and version.

    Parameters
    ----------
    reader_cls : class
        The reader class to register. Must have dm_type and version attributes.

    Returns
    -------
    class
        The registered reader class.
    """
    dm_type = reader_cls.dm_type
    version = reader_cls.version
    _HANDLERS[(dm_type, version)] = reader_cls
    return reader_cls


@_register_dmsy_reader
class DMSYDiscreteHandlerV1:
    """Handler for discrete decision matrices in DMSY format version 1.

    Parameters
    ----------
    dm_type : str
        The decision matrix type this handler supports.
    version : int
        The DMSY format version this handler supports.

    """

    dm_type = "discrete-dm"
    version = 1

    def to_dm(self, dm_data):
        """Convert DMSY data to DecisionMatrix object.

        Parameters
        ----------
        dm_data : dict
            Dictionary containing the decision matrix data from DMSY format.

        Returns
        -------
        DecisionMatrix
            The constructed DecisionMatrix object.
        """
        dm_data.pop("extra", None)
        dm = mkdm(**dm_data)
        return dm

    def to_yml(self, dm):
        """Convert DecisionMatrix object to DMSY-compatible data.

        Parameters
        ----------
        dm : DecisionMatrix
            The DecisionMatrix object to convert.

        Returns
        -------
        dict
            Dictionary containing the data in DMSY format.
        """
        dm_data = dm.to_dict()

        # Fix dtypes and add missing fields
        dm_data["dtypes"] = [dtype.name for dtype in dm_data["dtypes"]]
        dm_data["extra"] = {}

        return dm_data


# =============================================================================
# LOAD FUNCTIONS
# =============================================================================


def _read_dmsy_buffer(fp):
    """Read DMSY data from a file buffer.

    Parameters
    ----------
    fp : file-like object
        File buffer containing DMSY data.

    Returns
    -------
    DecisionMatrix
        The loaded DecisionMatrix object.

    Raises
    ------
    KeyError
        If no handler is found for the specified dm_type and version.
    """
    dmsy_data = yaml.safe_load(fp)
    dmsy_info = dmsy_data["dmsy"]
    dm_type = dmsy_info["dm_type"]
    dmsy_version = dmsy_info["version"]
    handler_cls = _HANDLERS[(dm_type, dmsy_version)]
    handler = handler_cls()
    dm = handler.to_dm(dmsy_data["data"])
    return dm


def read_dmsy(filepath_or_buffer):
    """Load a DecisionMatrix from a DMSY format file or buffer.

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        Path to the DMSY file or a file-like object containing DMSY data.

    Returns
    -------
    DecisionMatrix
        The loaded DecisionMatrix object.

    Examples
    --------
    >>> import skcriteria as skc
    >>> dm = skc.io.read_dmsy("dataset.dmsy")
    """
    if isinstance(filepath_or_buffer, (str, pathlib.Path)):
        with open(filepath_or_buffer, "r") as fp:
            return _read_dmsy_buffer(fp)
    return _read_dmsy_buffer(filepath_or_buffer)


# =============================================================================
# SAVE FUNCTIONS
# =============================================================================


def _get_metadata():
    """Generate metadata for DMSY files.

    Returns
    -------
    dict
        Dictionary containing metadata information including creation
        timestamp.

    """
    metadata = _DMSY_METADATA_DEFAULT_TEMPLATE.copy()
    metadata["created_at"] = dt.datetime.utcnow()
    return metadata


def _save_dmsy_buffer(dm, fp):
    """Save DecisionMatrix to a DMSY format buffer.

    Parameters
    ----------
    dm : DecisionMatrix
        The DecisionMatrix object to save.
    fp : file-like object
        File buffer to write the DMSY data to.

    Returns
    -------
    file-like object
        The file buffer that was written to.
    """
    dm_type, dmsy_version = DEFAULT_DM_TYPE, DEFAULT_DMSY_VERSION
    handler_cls = _HANDLERS[(dm_type, dmsy_version)]
    handler = handler_cls()
    dm_data = handler.to_yml(dm)

    dmsy_data = {
        "dmsy": {
            "dm_type": dm_type,
            "version": dmsy_version,
            "meta": _get_metadata(),
        },
        "data": dm_data,
    }

    yaml.dump(
        dmsy_data,
        fp,
        Dumper=CustomYAMLDumper,
        sort_keys=False,
        indent=2,
        default_flow_style=False,  # Only for dictionaries
    )
    return fp


def to_dmsy(dm, filepath_or_buffer):
    """Save a DecisionMatrix to a DMSY format file or buffer.

    Parameters
    ----------
    dm : DecisionMatrix
        The DecisionMatrix object to save.
    filepath_or_buffer : str or file-like object
        Path where to save the DMSY file or a file-like object to write to.

    Examples
    --------
    >>> import skcriteria as skc
    >>> dm = skc.mkdm([[1, 2], [3, 4]], [max, min])
    >>> skc.io.to_dmsy(dm, "output.dmsy")
    """
    if isinstance(filepath_or_buffer, (str, pathlib.Path)):
        with open(filepath_or_buffer, "w") as fp:
            return _save_dmsy_buffer(dm, fp)
    return _save_dmsy_buffer(dm, filepath_or_buffer)
