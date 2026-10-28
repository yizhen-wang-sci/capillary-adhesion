"""Recipe semantics: merging the config files a run is specified by, and addressing into them.

A run is specified by `params.toml` plus overlays, later overriding earlier. `[[batch]]` names
its axes by dotted path.
"""

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
        dict: The merged configuration.

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


def value_at_path(config: dict, path: str):
    """The value a dotted path names, e.g. ``"capillary.contact_angle_degree"``.

    Args:
        config: The configuration to read.
        path: Keys joined by ".", the same spelling a ``[[batch]]`` uses.

    Returns:
        Any: What sits there.

    Raises:
        KeyError: If the path leads through a key the config does not have.
    """
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base.

    Args:
        base: Values to start from. Not modified.
        override: Values taking precedence. A key holding a dict on both sides is merged
            key by key; anything else replaces what base holds.

    Returns:
        dict: A new dict. The sub-dicts held on both sides are new as well, everything
            else is shared with the inputs.
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
