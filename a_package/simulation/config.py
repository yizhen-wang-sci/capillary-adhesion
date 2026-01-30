"""
Configuration loading utilities.

Provides:
- load_config: Load and merge TOML files into plain dict
- backfill: Fill missing fields from defaults dict
- save_config: Persist config dict to TOML
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
    """Load and merge TOML config files. Later files override earlier."""
    if not paths:
        raise ValueError("At least one config path required")

    merged: dict[str, Any] = {}
    for path in paths:
        with open(Path(path), "rb") as fp:
            merged = _deep_merge(merged, tomllib.load(fp))
    return merged


def backfill_config(config: dict, defaults: dict):
    """Backfill missing fields in config from defaults. Config values take precedence."""
    return _deep_merge(defaults, config)


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict[str, Any], path: str | Path):
    """Save config dict to TOML file."""
    with open(Path(path), "wb") as fp:
        tomli_w.dump(config, fp)
