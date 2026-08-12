"""A record's input kept as a TOML file."""

import os
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w


def read_input(path: str | os.PathLike) -> dict:
    """Read the request a record was run against."""
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def write_input(path: str | os.PathLike, config: dict[str, Any]):
    """Write the request a record is to be run against, overwriting what is there."""
    with open(path, "wb") as fp:
        tomli_w.dump(config, fp)
