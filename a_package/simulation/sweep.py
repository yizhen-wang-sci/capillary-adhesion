"""
Parameter sweep expansion for parametric exploration.

Expands sweep specifications into parameter override dictionaries,
and applies overrides to Config objects.
"""

import copy
import itertools
import dataclasses as dc
from typing import Any, Iterator

import numpy as np

from .parameter import Config


@dc.dataclass
class SweepItem:
    """
    Single sweep parameter specification.

    Attributes
    ----------
    path : str
        Dot-notation path to config value (e.g., "problem.capillary.contact_angle_degree")
    values : list[Any]
        Expanded list of values to sweep over.
    """
    path: str
    values: list[Any]

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "SweepItem":
        """
        Create SweepItem from raw dict specification.

        Parameters
        ----------
        spec : dict
            Must contain 'path' and one of:
            - 'linspace': [start, stop, num]
            - 'logspace': [start, stop, num]
            - 'values': [v1, v2, ...]
        """
        path = spec["path"]

        if "linspace" in spec:
            start, stop, num = spec["linspace"]
            values = np.linspace(start, stop, int(num)).tolist()
        elif "logspace" in spec:
            start, stop, num = spec["logspace"]
            values = np.logspace(start, stop, int(num)).tolist()
        elif "values" in spec:
            values = list(spec["values"])
        else:
            raise ValueError(
                f"Sweep at path '{path}' has no values specified. "
                "Use linspace, logspace, or values."
            )

        return cls(path=path, values=values)


# -----------------------------------------------------------------------------
# Low-level: sweep spec -> override dicts
# -----------------------------------------------------------------------------

def expand_sweep_spec(sweep_spec: list[dict] | None) -> Iterator[dict[str, Any]]:
    """
    Expand sweep specification into parameter override dicts.

    Parameters
    ----------
    sweep_spec : list[dict] | None
        List of sweep definitions, each with 'path' and values spec
        (linspace, logspace, or values).

    Yields
    ------
    dict[str, Any]
        Dict mapping path -> value for each combination.
        Empty dict if no sweeps defined.

    Examples
    --------
    >>> spec = [{"path": "a.b", "values": [1, 2]}]
    >>> list(expand_sweep_spec(spec))
    [{"a.b": 1}, {"a.b": 2}]
    """
    if not sweep_spec:
        yield {}
        return

    items = [SweepItem.from_dict(spec) for spec in sweep_spec]

    paths = [item.path for item in items]
    value_lists = [item.values for item in items]

    for combo in itertools.product(*value_lists):
        yield dict(zip(paths, combo))


def count_sweep_combinations(sweep_spec: list[dict] | None) -> int:
    """
    Count the total number of configurations that would be generated.

    Returns 1 if no sweeps are defined.
    """
    if not sweep_spec:
        return 1

    items = [SweepItem.from_dict(spec) for spec in sweep_spec]
    total = 1
    for item in items:
        total *= len(item.values)
    return total


# -----------------------------------------------------------------------------
# High-level: Config -> list[Config]
# -----------------------------------------------------------------------------

def _set_nested_value(config: Config, path: str, value: Any) -> None:
    """
    Set a nested value using dot notation.

    The first part of the path is a Config attribute (domain, problem, etc.),
    the rest navigates through nested dicts.

    Example: path="problem.capillary.contact_angle_degree" sets
             config.problem["capillary"]["contact_angle_degree"] = value
    """
    parts = path.split(".")
    # First part is a Config attribute
    obj = getattr(config, parts[0])
    # Navigate through nested dicts
    for part in parts[1:-1]:
        obj = obj[part]
    # Set the final value
    obj[parts[-1]] = value


def _apply_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """
    Apply path->value overrides to a config.

    Parameters
    ----------
    config : Config
        Original configuration.
    overrides : dict[str, Any]
        Dict mapping dot-notation paths to values.

    Returns
    -------
    Config
        New config with overrides applied and sweep cleared.
    """
    if not overrides:
        return dc.replace(config, sweep=[])

    new_config = copy.deepcopy(config)
    for path, value in overrides.items():
        _set_nested_value(new_config, path, value)
    return dc.replace(new_config, sweep=[])


def expand_configs(config: Config) -> list[Config]:
    """
    Expand config with sweeps into list of individual configs.

    Parameters
    ----------
    config : Config
        Configuration, possibly with sweep definitions.

    Returns
    -------
    list[Config]
        List of expanded configs (sweep field cleared).
    """
    return [
        _apply_overrides(config, overrides)
        for overrides in expand_sweep_spec(config.sweep)
    ]
