"""Rough surfaces cycled through approach and retraction, at several contact
angles: how wetting affinity changes the adhesion hysteresis.
"""

import sys as _sys
from pathlib import Path as _Path

# Make _common/modules importable
_sys.path.insert(
    0,
    str(next(d for d in _Path(__file__).resolve().parents if (d / "_common").is_dir())),
)

# ruff: disable[F401]
from _common.bases import SPATIAL_BASES, save_grid, element_values
from _common.capillary import build_phase_mixture
from _common.cli import cli_config, cli_records
from _common.config import load_config
from _common.constraint import build_liquid_volume
from _common.grid import build_grid
from _common.init_guess import square_init_guess
from _common.level_set import solve_phase_by_level_set
from _common.console import CONSOLE_LOGGERS, setup_console
from _common.optimizer import build_optimizer
from _common.frontier import count_complete_points, seed_point_scalars
from _common.surface import face_params, build_surface, report_surface_stats
from _common.term import Term
from _common.trajectory_separation import build_trajectory, preview_surface_approaching

RECORD_NAME_PATHS = {"theta": "capillary.contact_angle_degree"}
"""Which config value each record-name field is taken from."""

RECORD_NAMING_TYPES = {"theta": float}
"""What each record-name field parses back as."""
