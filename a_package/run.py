"""
Black-box simulation execution.

Provides run_simulation() and run_sweep() - config in, results out.
Helper functions translate config to primitives.
"""

import logging
from typing import Any

import numpy as np
import numpy.random as random

from a_package.config import Config, save_config, expand_sweeps, count_sweep_combinations
from a_package.domain import Grid
from a_package.problem import generate_surface, NodalFormCapillary
from a_package.simulation import Simulation, SimulationIO
from a_package.runtime import RunDir, CaseDir, switch_log_file


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers: config -> primitives
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


def compute_liquid_volume(
    grid: Grid,
    constraint_cfg: dict[str, Any],
    upper: np.ndarray,
    lower: np.ndarray,
    capillary_args: dict[str, Any],
    trajectory: np.ndarray,
) -> float:
    """
    Compute liquid volume from percentage specification.

    The percentage is relative to the maximum possible volume
    (full liquid at minimum separation).
    """
    # Create formulation at minimum separation to compute reference volume
    formulation = NodalFormCapillary(grid, capillary_args)
    z_min = np.amin(trajectory)
    gap = np.clip(upper + z_min - lower, 0, None)
    formulation.set_gap(gap)

    # Full liquid phase field
    full_liquid = np.ones([1, 1, *grid.nb_elements])
    formulation.set_phase(full_liquid)

    # Compute volume as percentage of full
    V_percent = 0.01 * constraint_cfg["liquid_volume_percent"]
    return formulation.get_volume() * V_percent


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_simulation(config: Config, run_dir: RunDir) -> SimulationIO:
    """
    Run a single simulation from config.

    This is the black-box interface: config in, results out.
    Dispatches to appropriate Simulation method based on constraint type.

    Parameters
    ----------
    config : Config
        Complete simulation configuration.
    run_dir : RunDir
        Run directory object with results_dir, parameters_dir, etc.

    Returns
    -------
    SimulationIO
        IO object for accessing saved results.
    """
    # Build primitives from config
    grid = create_grid_from_config(config)
    upper = generate_surface_from_config(grid, config.problem["upper"])
    lower = generate_surface_from_config(grid, config.problem["lower"])
    capillary_args = build_capillary_args(config)
    solver_args = build_solver_args(config)
    trajectory = build_trajectory(config)

    # Random initial phase field
    rng = random.default_rng()
    phase_init = rng.random((1, 1, *grid.nb_elements))

    # Create simulation object
    logger.info(f"Starting simulation with output to {run_dir.results_dir}")
    simulation = Simulation(grid, capillary_args, solver_args)

    # Dispatch based on constraint type
    constraint_cfg = config.simulation["constraint"]
    constraint_type = constraint_cfg["type"]

    if constraint_type == "constant_volume":
        volume = compute_liquid_volume(
            grid, constraint_cfg, upper, lower, capillary_args, trajectory
        )
        return simulation.run_with_constant_volume(
            upper, lower, trajectory, volume, run_dir.results_dir, phase_init=phase_init
        )
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def run_sweep(config: Config, case_dir: CaseDir) -> list[SimulationIO]:
    """
    Run simulation(s) from config, handling sweeps if present.

    If config has no sweeps, runs single simulation.
    If config has sweeps, creates all runs upfront and runs each.

    Parameters
    ----------
    config : Config
        Configuration, possibly with sweep definitions.
    case_dir : CaseDir
        Data directory for this case.

    Returns
    -------
    list[SimulationIO]
        List of IO objects, one per run.
    """
    nb_configs = count_sweep_combinations(config)

    if nb_configs == 1:
        # No sweeps, single run
        run_dir = case_dir.create_run(__file__, with_hash=True)
        switch_log_file(run_dir.log_file)
        return [run_simulation(config, run_dir)]

    # Parameter sweep - create all run dirs upfront
    run_dirs = case_dir.create_sweep(__file__, nb_configs, with_hash=True)

    results = []
    for index, (run_dir, expanded_config) in enumerate(zip(run_dirs, expand_sweeps(config))):
        switch_log_file(run_dir.log_file)
        logger.info(f"Run #{index + 1} of {nb_configs} runs.")

        # Save expanded config for reproducibility
        save_config(expanded_config, run_dir.parameters_dir / "config.toml")

        io = run_simulation(expanded_config, run_dir)
        results.append(io)

    return results
