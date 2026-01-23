"""
Configuration helpers for simulation scripts.

Provides shared utilities for translating config to primitives,
used by run_constant_volume.py and run_constant_pressure.py.
"""

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from a_package.simulation import Config, CaseDir, RunDir, switch_log_file
from a_package.domain import Grid, adapt_shape
from a_package.model import generate_surface


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config -> Primitives translation
# -----------------------------------------------------------------------------

def create_grid_from_config(config: Config) -> Grid:
    """Create a Grid from configuration."""
    grid_cfg = config.domain["grid"]
    a = grid_cfg["pixel_size"]
    N = grid_cfg["nb_pixels"]
    L = a * N
    return Grid([L, L], [N, N])


def get_surface_shape(config: Config, which: str) -> str:
    """
    Get the shape name for a surface.

    Parameters
    ----------
    config : Config
        The configuration object.
    which : str
        Either "upper" or "lower".

    Returns
    -------
    str
        The surface shape name.
    """
    return config.problem[which]["shape"]


def generate_surface_from_config(grid: Grid, surface_cfg: dict[str, Any]) -> np.ndarray:
    """
    Generate a surface from configuration dict.

    Extracts shape and passes remaining params to generate_surface.
    """
    cfg = dict(surface_cfg)  # copy to avoid mutation
    shape = cfg.pop("shape")
    return generate_surface(grid, shape, **cfg)


def build_capillary_args(config: Config) -> dict[str, Any]:
    """
    Build capillary model arguments from configuration.

    Translates user-facing config parameters to physics class parameters:
    - contact_angle_degree -> theta (radians)
    - interface_thickness -> eta
    """
    capillary = config.problem["capillary"]
    theta = (np.pi / 180) * capillary["contact_angle_degree"]
    eta = capillary["interface_thickness"]
    return {"eta": eta, "theta": theta}


def build_solver_args(config: Config) -> dict[str, Any]:
    """
    Build solver arguments from configuration.

    Translates user-facing config parameters to solver class parameters:
    - max_nb_iters -> max_inner_iter
    - max_nb_loops -> max_outer_loop
    - tol_constraints -> tol_constraint
    """
    optimizer = config.solver["optimizer"]
    return {
        "max_inner_iter": optimizer["max_nb_iters"],
        "max_outer_loop": optimizer["max_nb_loops"],
        "tol_convergence": optimizer["tol_convergence"],
        "tol_constraint": optimizer["tol_constraints"],
        "init_penalty_weight": optimizer["init_penalty_weight"],
    }


def build_trajectory(config: Config) -> np.ndarray:
    """Build separation trajectory from configuration."""
    traj_cfg = config.simulation["trajectory"]
    traj_type = traj_cfg["type"]

    if traj_type == "approach_retract":
        d_min = traj_cfg["min_separation"]
        d_max = traj_cfg["max_separation"]
        d_step = traj_cfg["step_size"]
        round_trip = traj_cfg.get("round_trip", True)

        nb_steps = round((d_max - d_min) / d_step) + 1
        # Start from max (approach), go to min
        separations = np.linspace(d_max, d_min, nb_steps)

        if round_trip:
            separations = np.concatenate([separations, np.flip(separations)[1:]])

        return separations

    elif traj_type == "explicit":
        return np.array(traj_cfg["values"])

    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


def inspect_config(config: Config) -> dict:
    """
    Extract computed primitives from config for inspection/preview.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    dict
        Dictionary with keys:
        - grid: Grid object
        - upper: upper surface height array
        - lower: lower surface height array
        - trajectory: separation values array
        - upper_shape: name of upper surface shape
        - lower_shape: name of lower surface shape
    """
    grid = create_grid_from_config(config)
    return {
        "grid": grid,
        "upper": generate_surface_from_config(grid, config.problem["upper"]),
        "lower": generate_surface_from_config(grid, config.problem["lower"]),
        "trajectory": build_trajectory(config),
        "upper_shape": get_surface_shape(config, "upper"),
        "lower_shape": get_surface_shape(config, "lower"),
    }


# -----------------------------------------------------------------------------
# Contact solver (moved from a_package/solver/contact.py)
# -----------------------------------------------------------------------------

class RigidContactSolver:
    """Computes the gap field between two surfaces at a given separation."""

    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def solve_gap(self, separation: float):
        return np.clip(separation + self.upper - self.lower, 0, None)


# -----------------------------------------------------------------------------
# Run staging (moved from a_package/simulation/staging.py)
# -----------------------------------------------------------------------------

def prepare_run(run_dir: RunDir) -> None:
    """
    Prepare infrastructure for a single run.

    Switches logging to the run's log file.
    """
    switch_log_file(run_dir.log_file)


def prepare_sweep(
    case_dir: CaseDir,
    nb_runs: int,
    script_path: str | Path,
) -> Iterator[RunDir]:
    """
    Create and prepare all run directories for a sweep.

    Handles both single-run and multi-run cases.
    Yields RunDir objects ready for use.

    Parameters
    ----------
    case_dir : CaseDir
        Case directory for this sweep.
    nb_runs : int
        Number of runs to create.
    script_path : str | Path
        Path to the calling script.

    Yields
    ------
    RunDir
        Run directory for each run.
    """
    if nb_runs == 1:
        # Single run
        run_dir = case_dir.create_run(script_path, with_hash=True)
        prepare_run(run_dir)
        yield run_dir
        return

    # Parameter sweep - create all run dirs upfront
    run_dirs = case_dir.create_sweep(script_path, nb_runs, with_hash=True)

    for index, run_dir in enumerate(run_dirs):
        logger.info(f"Run #{index + 1} of {nb_runs}")
        prepare_run(run_dir)
        yield run_dir
