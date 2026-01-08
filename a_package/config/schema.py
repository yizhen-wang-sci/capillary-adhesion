"""
Configuration schema.

Minimal schema that mirrors subpackage organization:
- domain: grid definitions
- problem: physics and geometry (upper, lower, capillary)
- solver: optimizer settings
- simulation: trajectory and constraint settings
- sweep: parameter sweep specifications

Each section is a raw dict - semantic knowledge lives in the consuming code
(run.py), not here. This avoids duplication between config schema and
problem/solver classes.
"""

import dataclasses as dc
from typing import Any


@dc.dataclass
class Config:
    """
    Top-level configuration.

    First-level keys match subpackage names for clarity.
    All sections are raw dicts to avoid schema duplication.
    """
    domain: dict[str, Any]
    problem: dict[str, Any]
    simulation: dict[str, Any]
    solver: dict[str, Any] = dc.field(default_factory=dict)
    sweep: list[dict[str, Any]] = dc.field(default_factory=list)
