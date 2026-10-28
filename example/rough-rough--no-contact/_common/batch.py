"""Batch expansion: the independent members one config file describes.

A batch is the set of records a recipe asks for: independent runs, one per combination. The
trajectory a single run walks lives in the `step` and `cycle` bases inside a record.

Typical usage example:
    An axis states `values`, `linspace` as start, stop, num, or `logspace` as 10^start to
    10^stop, num. The three below give 3 * 5 * 4 = 60 combinations.

        [[batch]]
        path = "problem.capillary.contact_angle"
        values = [30, 60, 90]

        [[batch]]
        path = "problem.upper.roughness.rms"
        linspace = [0.1, 1.0, 5]

        [[batch]]
        path = "solver.tolerance"
        logspace = [-6, -3, 4]

Note:
    A `[[batch]]` in an overlay replaces the whole list rather than merging into it.
"""

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np


def size_of_batch(config: dict) -> int:
    """How many configs the batch would yield.

    Args:
        config: The configuration. Not modified, its "batch" key stays in place.

    Returns:
        int: Size of the Cartesian product over the axes, 1 if no batch is defined.

    Raises:
        ValueError: If two axes share a path, or an axis has no value specification.
    """
    batch_spec = _concretize(config.get("batch", []))
    if len(batch_spec) == 0:
        return 1
    size = 1
    for values in batch_spec.values():
        size *= len(values)
    return size


def unroll_batch(config: dict) -> Iterator[dict]:
    """Iterate over the batch's parameter combinations.

    Args:
        config: The configuration. Its "batch" key is popped, and the varied paths are
            written in place.

    Yields:
        dict: One combination, in the same dict object re-mutated per iteration. A config with
            no batch is yielded once, unchanged.

    Raises:
        ValueError: If two axes share a path, or an axis has no value specification.
        KeyError: If a varied path leads through a key the config does not have.
    """
    batch_spec = _concretize(config.pop("batch", []))
    for update in _iter_updates(batch_spec):
        for path, value in update:
            _set_nested(config, path, value)
        yield config


def _concretize(axes: list[dict]):
    """Expand linspace/logspace into explicit values, and merge the axes into one dict.

    Args:
        axes: The batch axes as read from config, each with a "path" and one of "values",
            "linspace" or "logspace".

    Returns:
        dict[str, list]: The values to vary, keyed by path.

    Raises:
        ValueError: If two axes share a path, or an axis carries none of the three value
            specifications.

    Example:
        Input:  [{"path": "a.b", "linspace": [0, 1, 3]}, ...]
        Output: {"a.b": [0.0, 0.5, 1.0], ...}
    """
    result = {}
    for axis in axes:
        path = axis["path"]
        if path in result:
            raise ValueError(f"Duplicated batch axes at path {path}")
        if "values" in axis:
            values = list(axis["values"])
        elif "linspace" in axis:
            start, stop, num = axis["linspace"]
            values = np.linspace(start, stop, int(num)).tolist()
        elif "logspace" in axis:
            start, stop, num = axis["logspace"]
            values = np.logspace(start, stop, int(num)).tolist()
        else:
            raise ValueError(
                f"Batch axis at path '{path}' has no supported value specification. Use linspace, logspace, or values."
            )
        result[path] = values
    return result


def _iter_updates(batch_specs: dict[str, list]):
    """The (path, value) pairs of each Cartesian-product combination.

    Args:
        batch_specs: The values to vary, keyed by path.

    Yields:
        zip: The paths zipped with one combination of their values. The last path varies
            fastest.

    Example:
        Input:  {"a.b": [1, 2], "c": [3, 4]}
        Yields: zip producing ("a.b", 1), ("c", 3)
                zip producing ("a.b", 1), ("c", 4)
                zip producing ("a.b", 2), ("c", 3)
                zip producing ("a.b", 2), ("c", 4)
    """
    for combo in itertools.product(*batch_specs.values()):
        yield zip(batch_specs.keys(), combo)


def _set_nested(config: dict, path: str, value: Any):
    """Set the value at a dotted path: "a.b.c" sets config["a"]["b"]["c"].

    Args:
        config: The configuration, modified in place.
        path: The keys to walk, separated by dots.
        value: The value to write.

    Raises:
        KeyError: If a key along the way is missing. Only the last key is created.
    """
    keys = path.split(".")
    obj = config
    for key in keys[:-1]:
        obj = obj[key]
    obj[keys[-1]] = value
