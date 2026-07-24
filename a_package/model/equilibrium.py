"""
Equilibrium formulations for capillary contact problems.
"""

from a_package.domain import Problem
from .capillary import CapillaryBridge


def formulate_constant_volume_phase_problem(capillary: CapillaryBridge, volume: float, explicit_phase_bounds: bool=True):
    """
    min energy(phase)
    s.t. volume(phase) == volume
    """

    # Exploit the linearity in the volume Jacobian
    args = dict(
        get_x=capillary.get_phase,
        set_x=capillary.set_phase,
        get_f=capillary.get_energy,
        get_f_Dx=capillary.get_energy_jacobian,
        A=capillary.get_volume_jacobian().ravel(),
        b=volume,
        is_zeroed=capillary.gap_is_closed,
        communicator=capillary.communicator)

    # Explicit boundaries in case feasibility must be enforced
    if explicit_phase_bounds:
        args.update(dict(x_lb=capillary.phase_lb, x_ub=capillary.phase_ub))
    return Problem(**args)


def formulate_constant_pressure_phase_problem(capillary: CapillaryBridge, pressure: float, explicit_phase_bounds: bool=True):
    """
    min energy(phase) - pressure * volume(phase)
    """

    def helmholtz_potential():
        return capillary.get_energy() - pressure * capillary.get_volume()

    def helmholtz_potential_jacobian():
        return capillary.get_energy_jacobian() - pressure * capillary.get_volume_jacobian()

    # Exploit the linearity in the volume Jacobian
    args = dict(
        get_x=capillary.get_phase,
        set_x=capillary.set_phase,
        get_f=helmholtz_potential,
        get_f_Dx=helmholtz_potential_jacobian,
        is_zeroed=capillary.gap_is_closed,
        communicator=capillary.communicator)

    # Explicit boundaries in case feasibility must be enforced
    if explicit_phase_bounds:
        args.update(dict(x_lb=capillary.phase_lb, x_ub=capillary.phase_ub))
    return Problem(**args)
