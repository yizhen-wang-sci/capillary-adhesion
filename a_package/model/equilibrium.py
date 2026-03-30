"""
Equilibrium solvers for capillary contact.
"""

from types import SimpleNamespace

import numpy as np
import numpy.random as random

from a_package.domain import Optimizer
from .capillary import NodalFormCapillary
from .contact import RigidContact


def solve_rigid_constant_volume(
        grid, upper, lower, separation, volume, capillary_args, solver_args, phase_init=None, pressure_init=0.0):
    """
    Solve equilibrium at constant volume.

    Returns (gap, phase, result).
    """
    contact = RigidContact(upper, lower)
    capillary = NodalFormCapillary(grid, capillary_args)
    optimizer = Optimizer(**solver_args)

    contact.set_mean_separation(separation)
    gap = contact.get_gap()
    capillary.set_gap(gap)

    # Build the capillary problem
    def volume_constraint():
        return capillary.get_volume() - volume

    volume_constraint_jacobian = capillary.get_volume_jacobian

    problem = SimpleNamespace(
        get_x=capillary.get_phase,
        set_x=capillary.set_phase,
        get_f=capillary.get_energy,
        get_f_Dx=capillary.get_energy_jacobian,
        get_g=volume_constraint,
        get_g_Dx=volume_constraint_jacobian,
        x_lb=capillary.phase_lb,
        x_ub=capillary.phase_ub,
    )

    # initial guess
    if phase_init is None:
        rng = random.default_rng()
        phase_init = rng.random((1, 1, *grid.nb_elements))
    init_shape = phase_init.shape

    result = optimizer.solve_minimisation(problem, x0=phase_init, lam0=-pressure_init)
    phase = np.reshape(result['primal'], init_shape)
    pressure = -result['dual']

    return gap, phase, result


def solve_rigid_constant_pressure(
        grid, upper, lower, separation, pressure, capillary_args, solver_args, phase_init=None):
    """
    Solve equilibrium at constant pressure.

    Returns (gap, phase, result).
    """
    contact = RigidContact(upper, lower)
    capillary = NodalFormCapillary(grid, capillary_args)
    solver = Optimizer(**solver_args)

    contact.set_mean_separation(separation)
    gap = contact.get_gap()
    capillary.set_gap(gap)

    # Build the capillary problem
    def helmholtz_potential():
        return capillary.get_energy() + pressure * capillary.get_volume()

    def helmholtz_potential_jacobian():
        return capillary.get_energy_jacobian() + pressure * capillary.get_volume_jacobian()

    problem = SimpleNamespace(
        get_x=capillary.get_phase,
        set_x=capillary.set_phase,
        get_f=helmholtz_potential,
        get_f_Dx=helmholtz_potential_jacobian,
        x_lb=capillary.phase_lb,
        x_ub=capillary.phase_ub,
    )

    # initial guess
    if phase_init is None:
        rng = random.default_rng()
        phase_init = rng.random((1, 1, *grid.nb_elements))
    init_shape = phase_init.shape

    result = solver.solve_minimisation(problem, x0=phase_init)
    phase = np.reshape(result['primal'], init_shape)

    return gap, phase, result
