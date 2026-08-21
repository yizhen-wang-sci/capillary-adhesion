"""Configuration loading utilities."""

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w


def load_config(*paths: str | Path):
    """Load and merge TOML config files.

    Args:
        *paths: The TOML files, where later files override earlier.

    Returns:
        The merged configuration.

    Raises:
        ValueError: If no path is given.
    """
    if not paths:
        raise ValueError("At least one config path required")

    merged: dict[str, Any] = {}
    for path in paths:
        with open(Path(path), "rb") as fp:
            merged = _deep_merge(merged, tomllib.load(fp))
    return merged


def backfill_config(config: dict, defaults: dict):
    """Backfill missing fields in config from defaults.

    Args:
        config: The configuration to fill in. Not modified.
        defaults: Values to fall back on, merged in at every depth.

    Returns:
        A new dict, suffixed with defaults.
    """
    return _deep_merge(defaults, config)


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base.

    Args:
        base: Values to start from. Not modified.
        override: Values taking precedence. A key holding a dict on both sides is merged key
            by key; anything else replaces what base holds.

    Returns:
        A new dict. The sub-dicts held on both sides are new as well, everything else is
        shared with the inputs.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict[str, Any], path: str | Path):
    """Save config dict to TOML file.

    Args:
        config: The configuration to write.
        path: Destination file, overwritten if it exists.
    """
    with open(Path(path), "wb") as fp:
        tomli_w.dump(config, fp)
