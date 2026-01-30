"""
Parameter sweep expansion for parametric exploration.

Supported sweep format in config TOML:

    [[sweep]]
    path = "problem.capillary.contact_angle"
    values = [30, 60, 90]

    [[sweep]]
    path = "problem.upper.roughness.rms"
    linspace = [0.1, 1.0, 5]  # start, stop, num

    [[sweep]]
    path = "solver.tolerance"
    logspace = [-6, -3, 4]  # 10^start to 10^stop, num points

Above generates 3 * 5 * 4 = 60 config combinations.
"""

import itertools
from typing import Any, Iterator

import numpy as np


def unroll_sweep(config: dict) -> Iterator[dict]:
    """
    Iterate over sweep parameter combinations.

    Pops "sweep" key from config and yields the same dict object
    with each parameter combination applied in place. Caller must
    copy if individual configs need to be preserved.

    If no sweep defined, yields config once unchanged.
    """
    sweep_spec = _concretize(config.pop("sweep", []))
    for update in _iter_updates(sweep_spec):
        for path, value in update:
            _set_nested(config, path, value)
        yield config


def _concretize(sweeps: list[dict]):
    """
    Expand linspace/logspace into explicit values list and merge sweeps into one specification dict

    Input:  [{"path": "a.b", "linspace": [0, 1, 3]}, ...]
    Output: {"a.b": [0.0, 0.5, 1.0], ...}
    """
    result = {}
    for sweep in sweeps:
        path = sweep["path"]
        if path in result:
            raise ValueError(f"Duplicated sweeps at path {path}")
        if "values" in sweep:
            values = list(sweep["values"])
        elif "linspace" in sweep:
            start, stop, num = sweep["linspace"]
            values = np.linspace(start, stop, int(num)).tolist()
        elif "logspace" in sweep:
            start, stop, num = sweep["logspace"]
            values = np.logspace(start, stop, int(num)).tolist()
        else:
            raise ValueError(
                f"Sweep at path '{path}' has no supported value specification. "
                "Use linspace, logspace, or values."
            )
        result[path] = values
    return result


def _iter_updates(sweep_specs: dict[str, list]):
    """
    Yield iterable of (path, value) pairs for each Cartesian product combination.

    Input:  {"a.b": [1, 2], "c": [3, 4]}
    Yields: zip producing ("a.b", 1), ("c", 3)
            zip producing ("a.b", 1), ("c", 4)
            zip producing ("a.b", 2), ("c", 3)
            zip producing ("a.b", 2), ("c", 4)
    """
    for combo in itertools.product(*sweep_specs.values()):
        yield zip(sweep_specs.keys(), combo) 


def _set_nested(config: dict, path: str, value: Any):
    """Set value at dot-notation path. E.g., "a.b.c" sets config["a"]["b"]["c"]."""
    keys = path.split(".")
    obj = config
    for key in keys[:-1]:
        obj = obj[key]
    obj[keys[-1]] = value
