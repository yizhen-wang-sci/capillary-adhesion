"""Equilibrium formulations for capillary contact problems."""

from a_package.domain import Problem, OptimizerResult
from .capillary import CapillaryBridge


def formulate_constant_volume_phase_problem(
    capillary: CapillaryBridge, volume: float, explicit_phase_bounds: bool = True
):
    """Minimise energy(phase) subject to volume(phase) == volume.

    Args:
        capillary: The physics model providing the energy and its Jacobian w.r.t. phase field.
        volume: The liquid volume to hold constant.
        explicit_phase_bounds: Whether to pass the phase bounds to the optimizer.

    Returns:
        An adapted problem the optimizer can handle, whose dual variable is the pressure.
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
        communicator=capillary.communicator,
    )

    # Explicit boundaries in case feasibility must be enforced
    if explicit_phase_bounds:
        args.update(dict(x_lb=capillary.phase_lb, x_ub=capillary.phase_ub))
    return Problem(**args)


def extract_pressure_in_constant_volume_solution(result: OptimizerResult):
    """Read the pressure out of a solved constant-volume problem.

    Args:
        result: From solving a problem built by `formulate_constant_volume_phase_problem`.

    Returns:
        The pressure, divided by the surface tension.
    """
    # NOTE: in NuMPI LinearConstraint, it defines lagrangian multiplier with "-lambda ...",
    # hence lambda and pressure have the same sign. For this problem, precisely,
    # lambda = pressure / surface tension
    pressure_per_surface_tension = result["dual"]
    return pressure_per_surface_tension


def formulate_constant_pressure_phase_problem(
    capillary: CapillaryBridge, pressure: float, explicit_phase_bounds: bool = True
):
    """Minimise energy(phase) - pressure * volume(phase).

    Args:
        capillary: The physics model providing the energy and its Jacobian w.r.t. phase field.
        pressure: The pressure to hold constant, in units of the surface tension.
        explicit_phase_bounds: Whether to pass the phase bounds to the optimizer.

    Returns:
        An adapted problem the optimizer can handle.
    """

    def helmholtz_potential():
        """Free energy of the capillary minus the work done against the constant pressure."""
        return capillary.get_energy() - pressure * capillary.get_volume()

    def helmholtz_potential_jacobian():
        """Derivative of `helmholtz_potential` with respect to the phase."""
        return capillary.get_energy_jacobian() - pressure * capillary.get_volume_jacobian()

    # Exploit the linearity in the volume Jacobian
    args = dict(
        get_x=capillary.get_phase,
        set_x=capillary.set_phase,
        get_f=helmholtz_potential,
        get_f_Dx=helmholtz_potential_jacobian,
        is_zeroed=capillary.gap_is_closed,
        communicator=capillary.communicator,
    )

    # Explicit boundaries in case feasibility must be enforced
    if explicit_phase_bounds:
        args.update(dict(x_lb=capillary.phase_lb, x_ub=capillary.phase_ub))
    return Problem(**args)
