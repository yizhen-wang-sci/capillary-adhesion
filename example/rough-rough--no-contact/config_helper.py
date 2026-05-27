"""
Configuration helpers for parallel_rough_rigid_contact simulations.
"""
from typing import Any

import numpy as np

from a_package.domain import Grid, ProjectedLbfgs
from a_package.model import PhaseMixture


__all__ = ['build_grid', 'build_phase_mixture', 'build_optimizer', 'build_trajectory']


def build_grid(config: dict):
    """Create a grid from configuration."""
    section = config["grid"]
    a = section["pixel_size"]
    N = section["nb_pixels"]
    L = a * N
    return Grid([N, N], [L, L])


def build_phase_mixture(config: dict):
    """Build capillary phase mixture from configuration."""
    section = config["capillary"]
    theta = (np.pi / 180) * section["contact_angle_degree"]
    eta = section["interface_thickness"]
    # epsilon = section["perimeter_weight"]
    return PhaseMixture(eta, theta)


def build_optimizer(config: dict):
    """Build solver from configuration."""
    section = config["optimizer"]
    return ProjectedLbfgs(max_inner_iter=section["max_nb_iters"], tol_gradient=section["tol_gradient"])


def build_trajectory(config: dict):
    """Build separation trajectory from configuration."""
    section = config["trajectory"]
    d_min = section["min_separation"]
    d_max = section["max_separation"]
    d_step = section["step_size"]
    round_trip = section.get("round_trip", True)

    nb_steps = round((d_max - d_min) / d_step) + 1
    separations = np.linspace(d_max, d_min, nb_steps)

    if round_trip:
        separations = np.concatenate([separations, np.flip(separations)[1:]])

    return separations
