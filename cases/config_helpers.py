"""
Configuration helpers for simulation scripts.

Provides shared utilities for translating config to primitives.
"""

import logging
from typing import Any

import numpy as np

from a_package.domain import Grid


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config -> Primitives translation
# -----------------------------------------------------------------------------

def create_grid_from_config(config: dict) -> Grid:
    """Create a Grid from configuration."""
    grid_cfg = config["domain"]["grid"]
    a = grid_cfg["pixel_size"]
    N = grid_cfg["nb_pixels"]
    L = a * N
    return Grid([L, L], [N, N])


def build_capillary_args(config: dict) -> dict[str, Any]:
    """
    Build capillary model arguments from configuration.

    Translates user-facing config parameters to physics class parameters:
    - contact_angle_degree -> theta (radians)
    - interface_thickness -> eta
    """
    capillary = config["problem"]["capillary"]
    theta = (np.pi / 180) * capillary["contact_angle_degree"]
    eta = capillary["interface_thickness"]
    return {"eta": eta, "theta": theta}


def build_solver_args(config: dict) -> dict[str, Any]:
    """
    Build solver arguments from configuration.

    Translates user-facing config parameters to solver class parameters:
    - max_nb_iters -> max_inner_iter
    - max_nb_loops -> max_outer_loop
    - tol_constraints -> tol_constraint
    """
    optimizer = config["solver"]["optimizer"]
    return {
        "max_inner_iter": optimizer["max_nb_iters"],
        "max_outer_loop": optimizer["max_nb_loops"],
        "tol_convergence": optimizer["tol_convergence"],
        "tol_constraint": optimizer["tol_constraints"],
        "init_penalty_weight": optimizer["init_penalty_weight"],
    }


def build_trajectory(config: dict) -> np.ndarray:
    """Build separation trajectory from configuration."""
    traj_cfg = config["simulation"]["trajectory"]
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
