"""
Configuration schema and TOML loading/saving.

Provides:
- Config dataclass for typed configuration
- TOML loading and saving utilities
"""

import sys
import dataclasses as dc
from pathlib import Path
from typing import Any

# Import tomllib or backport
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

@dc.dataclass
class Config:
    """
    Top-level configuration.

    First-level keys match subpackage names for clarity.
    All sections are raw dicts to avoid schema duplication.

    Attributes
    ----------
    domain : dict
        Grid definitions (pixel_size, nb_pixels).
    problem : dict
        Physics and geometry (upper, lower, capillary).
    simulation : dict
        Trajectory and constraint settings.
    solver : dict
        Optimizer settings.
    sweep : list[dict]
        Parameter sweep specifications.
    """
    domain: dict[str, Any]
    problem: dict[str, Any]
    simulation: dict[str, Any]
    solver: dict[str, Any] = dc.field(default_factory=dict)
    sweep: list[dict[str, Any]] = dc.field(default_factory=list)


# -----------------------------------------------------------------------------
# Loading and Saving
# -----------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, with override taking precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(*paths: str | Path) -> Config:
    """
    Load TOML configuration file(s) and return a Config object.

    When multiple paths are provided, files are loaded in order and
    later files override values from earlier files (deep merge).
    """
    if not paths:
        raise ValueError("At least one config path required")

    # Load and merge all files
    merged: dict[str, Any] = {}
    for path in paths:
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)
        merged = _deep_merge(merged, data)

    # Extract sweeps (TOML uses [[sweep]] array syntax)
    sweep = merged.pop("sweep", [])

    return Config(
        domain=merged.get("domain", {}),
        problem=merged.get("problem", {}),
        solver=merged.get("solver", {}),
        simulation=merged.get("simulation", {}),
        sweep=sweep,
    )


def save_config(config: Config, path: str | Path) -> None:
    """Save a Config object to a TOML file."""
    path = Path(path)
    data: dict[str, Any] = {}

    if config.domain:
        data["domain"] = config.domain
    if config.problem:
        data["problem"] = config.problem
    if config.solver:
        data["solver"] = config.solver
    if config.simulation:
        data["simulation"] = config.simulation
    if config.sweep:
        data["sweep"] = config.sweep

    with open(path, "wb") as f:
        tomli_w.dump(data, f)
