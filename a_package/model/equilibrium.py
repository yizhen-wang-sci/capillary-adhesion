"""
Equilibrium formulations for capillary contact problems.
"""

from a_package.domain import Problem
from .capillary import CapillaryBridge


def formulate_constant_volume_phase_problem(capillary: CapillaryBridge, volume: float):
    """
    min energy(phase)
    s.t. volume(phase) == volume
    """

    # def volume_constraint():
    #     return capillary.get_volume() - volume
    #
    # volume_constraint_jacobian = capillary.get_volume_jacobian
    #
    # return Problem(get_x=capillary.get_phase,
    #                set_x=capillary.set_phase,
    #                get_f=capillary.get_energy,
    #                get_f_Dx=capillary.get_energy_jacobian,
    #                get_g=volume_constraint,
    #                get_g_Dx=volume_constraint_jacobian,
    #                x_lb=capillary.phase_lb,
    #                x_ub=capillary.phase_ub)

    # Exploit the linearity in the volume Jacobian
    return Problem(get_x=capillary.get_phase,
                   set_x=capillary.set_phase,
                   get_f=capillary.get_energy,
                   get_f_Dx=capillary.get_energy_jacobian,
                   A=capillary.get_volume_jacobian().ravel(),
                   b=volume,
                   x_lb=capillary.phase_lb,
                   x_ub=capillary.phase_ub,
                   is_zeroed=capillary.gap_is_closed,
                   communicator=capillary.communicator)


def formulate_constant_pressure_phase_problem(capillary: CapillaryBridge, pressure: float):
    """
    min energy(phase) - pressure * volume(phase)
    """

    def helmholtz_potential():
        return capillary.get_energy() - pressure * capillary.get_volume()

    def helmholtz_potential_jacobian():
        return capillary.get_energy_jacobian() - pressure * capillary.get_volume_jacobian()

    return Problem(get_x=capillary.get_phase,
                   set_x=capillary.set_phase,
                   get_f=helmholtz_potential,
                   get_f_Dx=helmholtz_potential_jacobian,
                   x_lb=capillary.phase_lb,
                   x_ub=capillary.phase_ub,
                   is_zeroed=capillary.gap_is_closed,
                   communicator=capillary.communicator)
