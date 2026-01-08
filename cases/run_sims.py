"""
Run simulation from config file.

Usage:
    python -m cases.run_config config.toml

Workflow:
    1. Load config
    2. Extract primitives via inspect_config() for naming/preview
    3. Derive case directory name from surface shapes
    4. Run simulation(s) via run_sweep()
    5. Generate animations for each result
"""

import logging
import os
import sys

from a_package.config import load_config
from a_package.runtime import CaseDir, reset_logging
from a_package.run import run_sweep, inspect_config

from cases.visualisation import create_overview_animation, preview_surface_and_gap


show_me_preview = False
logger = logging.getLogger(__name__)


def main():
    reset_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m cases.load_unload config.toml")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # Extract primitives for inspection/naming
    primitives = inspect_config(config)

    # visual check
    if show_me_preview:
        preview_surface_and_gap(primitives)

    # setup case directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    shape_name = f'{primitives["upper_shape"]}-on-{primitives["lower_shape"]}'
    base_dir = os.path.join(script_name, shape_name)
    case_dir = CaseDir(base_dir)

    # Run simulation(s) - handles sweeps automatically
    ios = run_sweep(config, case_dir)

    # Create visualisations
    for io in ios:
        create_overview_animation(io, io.grid)


if __name__ == "__main__":
    main()
